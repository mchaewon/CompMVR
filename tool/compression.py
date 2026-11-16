import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

try:
    from tqdm import tqdm
except ImportError:
    import subprocess as _sp
    _sp.run(["pip", "install", "tqdm", "-q"])
    from tqdm import tqdm


FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")

INPUT_DIR = r"/NIPS/train_138+YUV"
OUTPUT_BASE_DIR = r"./trainset"
QP_VALUES = list(range(50, 19, -2))
N_WORKERS = 4
FPS = 30
GOP_SIZE = 32
B_FRAMES = 3

CODECS_ENABLED = {
    "AVC": True,
    "HEVC": True,
    "AV1": True,
    "VP9": True,
}

BITSTREAM_EXT = {
    "AVC": ".mp4",
    "HEVC": ".mp4",
    "AV1": ".mkv",
    "VP9": ".webm",
}


def expected_yuv420p_size(width, height, frames):
    return width * height * 3 // 2 * frames


def _run_command(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            lf.write(line)
        proc.wait()
    return proc.returncode


def _tail_log(log_path, n=30):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except Exception:
        return ""


def make_output_dirs(output_base, codec, scene_name, qp):
    mid_base = os.path.join(output_base, f"{codec}_{scene_name}_QP{qp}")
    comp_dir = os.path.join(mid_base, f"comp_result_QP{qp}")

    os.makedirs(comp_dir, exist_ok=True)
    return comp_dir


def is_completed_job(output_base, codec, scene_name, qp, base_name, width, height, frames):
    comp_dir = make_output_dirs(output_base, codec, scene_name, qp)
    bitstream_path = os.path.join(comp_dir, f"{base_name}{BITSTREAM_EXT[codec]}")
    recon_path = os.path.join(comp_dir, f"{base_name}.yuv")

    if not os.path.exists(bitstream_path) or not os.path.exists(recon_path):
        return False
    if os.path.getsize(bitstream_path) <= 0:
        return False

    expected_size = expected_yuv420p_size(width, height, frames)
    try:
        recon_size = os.path.getsize(recon_path)
    except OSError:
        return False
    return recon_size == expected_size


def _software_encoder(codec):
    return {
        "AVC": "libx264",
        "HEVC": "libx265",
        "AV1": "libsvtav1",
        "VP9": "libvpx-vp9",
    }[codec]


def _hardware_encoder(codec, backend):
    mapping = {
        "nvenc": {
            "AVC": "h264_nvenc",
            "HEVC": "hevc_nvenc",
            "AV1": "av1_nvenc",
        },
        "qsv": {
            "AVC": "h264_qsv",
            "HEVC": "hevc_qsv",
            "AV1": "av1_qsv",
            "VP9": "vp9_qsv",
        },
        "vaapi": {
            "AVC": "h264_vaapi",
            "HEVC": "hevc_vaapi",
            "AV1": "av1_vaapi",
            "VP9": "vp9_vaapi",
        },
    }
    enc = mapping.get(backend, {}).get(codec)
    if enc is None:
        raise ValueError(f"{backend} backend does not support {codec}")
    return enc


def resolve_encoder(codec, backend):
    if backend == "software":
        return _software_encoder(codec)
    return _hardware_encoder(codec, backend)


def build_ra_video_options(codec, encoder, backend, qp, gop_size, b_frames):
    fixed_gop_opts = [
        "-g", str(gop_size),
        "-keyint_min", str(gop_size),
        "-sc_threshold", "0",
    ]

    if backend == "software":
        if codec == "AVC":
            opts = fixed_gop_opts + [
                "-bf", str(b_frames),
                "-qp", str(qp),
                "-x264-params",
                (
                    f"keyint={gop_size}:min-keyint={gop_size}:"
                    f"scenecut=0:open-gop=0:bframes={b_frames}"
                ),
            ]
        elif codec == "HEVC":
            opts = fixed_gop_opts + [
                "-bf", str(b_frames),
                "-qp", str(qp),
                "-x265-params",
                (
                    f"keyint={gop_size}:min-keyint={gop_size}:"
                    f"scenecut=0:open-gop=0:bframes={b_frames}"
                ),
            ]
        elif codec == "AV1":
            # SVT-AV1 does not use H.26x-style B-frame options. Use a fixed
            # keyframe interval with scene detection disabled, and keep lookahead
            # available so the encoder can build its hierarchical prediction tree.
            opts = fixed_gop_opts + [
                "-qp", str(qp),
                "-preset", "6",
                "-svtav1-params",
                f"keyint={gop_size}:scd=0:irefresh-type=2:hierarchical-levels=4",
            ]
        elif codec == "VP9":
            # VP9 has no H.26x-style B-frame RA mode. This enforces a fixed
            # random-access keyframe interval without scene-cut keyframes.
            opts = fixed_gop_opts + [
                "-crf", str(qp),
                "-b:v", "0",
                "-row-mt", "1",
                "-tile-columns", "2",
                "-aq-mode", "0",
                "-lag-in-frames", "25",
                "-auto-alt-ref", "1",
            ]
        return opts

    if backend == "nvenc":
        opts = fixed_gop_opts
        if codec in {"AVC", "HEVC"}:
            opts += ["-bf", str(b_frames), "-forced-idr", "1"]
        if codec == "AV1":
            # Explicit AV1 quality mapping for av1_nvenc.
            opts += ["-rc", "constqp", "-qp", str(qp), "-preset", "p5"]
        else:
            opts += ["-rc", "constqp", "-qp", str(qp), "-preset", "p5"]
        return opts

    if backend == "qsv":
        opts = fixed_gop_opts
        if codec in {"AVC", "HEVC"}:
            opts += ["-bf", str(b_frames)]
        if codec == "AV1":
            # AV1 QSV quality is controlled by global_quality.
            opts += ["-global_quality", str(qp), "-look_ahead", "0"]
        else:
            opts += ["-global_quality", str(qp), "-look_ahead", "0"]
        return opts

    if backend == "vaapi":
        opts = fixed_gop_opts
        if codec in {"AVC", "HEVC"}:
            opts += ["-bf", str(b_frames)]
        if codec == "AV1":
            # Explicit AV1 quality mapping for av1_vaapi.
            opts += ["-qp", str(qp)]
        else:
            opts += ["-qp", str(qp)]
        return opts

    raise ValueError(f"Unsupported backend: {backend}")


def build_input_args(input_path, width, height, frames, backend, hw_device):
    args = [
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s:v", f"{width}x{height}",
        "-r", str(FPS),
        "-i", input_path,
        "-frames:v", str(frames),
    ]

    if backend == "vaapi":
        if not hw_device:
            raise ValueError("vaapi backend requires --hw-device (e.g. /dev/dri/renderD128)")
        return ["-vaapi_device", hw_device] + args
    return args


def build_filter_args(backend):
    if backend == "vaapi":
        return ["-vf", "format=nv12,hwupload"]
    return []


def build_encode_cmd(
    input_path,
    width,
    height,
    frames,
    qp,
    codec,
    bitstream_path,
    vstats_path,
    backend,
    hw_device,
):
    encoder = resolve_encoder(codec, backend)
    cmd = [FFMPEG_PATH]
    cmd += build_input_args(input_path, width, height, frames, backend, hw_device)
    cmd += build_filter_args(backend)
    cmd += ["-c:v", encoder]
    cmd += build_ra_video_options(codec, encoder, backend, qp, GOP_SIZE, B_FRAMES)
    cmd += ["-vstats_file", vstats_path, "-vstats_version", "2"]
    cmd += ["-an", bitstream_path]
    return cmd, encoder


def build_decode_cmd(bitstream_path, recon_path, width, height, frames):
    return [
        FFMPEG_PATH,
        "-y",
        "-i", bitstream_path,
        "-frames:v", str(frames),
        "-pix_fmt", "yuv420p",
        "-f", "rawvideo",
        recon_path,
    ]


def parse_vstats_qps(vstats_path):
    frames = []
    if not os.path.exists(vstats_path):
        return frames

    frame_re = re.compile(r"\bframe=\s*(\d+)")
    qp_re = re.compile(r"\bq=\s*([\-\d.]+)")
    type_re = re.compile(r"\btype=\s*([IPB])")
    time_re = re.compile(r"\btime=\s*([\d:.+-]+)")

    try:
        with open(vstats_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                frame_match = frame_re.search(line)
                qp_match = qp_re.search(line)
                if not frame_match or not qp_match:
                    continue
                type_match = type_re.search(line)
                time_match = time_re.search(line)
                frames.append(
                    {
                        "index": int(frame_match.group(1)),
                        "pts_time": time_match.group(1) if time_match else None,
                        "pict_type": type_match.group(1) if type_match else None,
                        "qp": float(qp_match.group(1)),
                    }
                )
    except Exception:
        return []

    return frames


def parse_encoder_log_qps(enc_log_path):
    # Try to heuristically extract QP numbers from encoder log (per-frame entries)
    qps = []
    if not os.path.exists(enc_log_path):
        return qps
    qp_patterns = [
        re.compile(r"\bQP[:=]\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bqp[:=]\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bq[:=]\s*(\d+)\b", re.IGNORECASE),
    ]
    try:
        with open(enc_log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for p in qp_patterns:
                    m = p.search(line)
                    if m:
                        try:
                            qps.append(int(m.group(1)))
                        except Exception:
                            pass
                        break
    except Exception:
        return []
    return qps


def save_qp_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def run_single_job(job):
    input_path = job["input_path"]
    scene_name = job["scene_name"]
    width = int(job["width"])
    height = int(job["height"])
    frames = int(job["frames"])
    codec = job["codec"]
    qp = int(job["qp"])
    output_base = job["output_base"]
    base_name = job["base_name"]
    backend = job["backend"]
    hw_device = job.get("hw_device")

    comp_dir = make_output_dirs(output_base, codec, scene_name, qp)
    bitstream_path = os.path.join(comp_dir, f"{base_name}{BITSTREAM_EXT[codec]}")
    recon_path = os.path.join(comp_dir, f"{base_name}.yuv")
    enc_log = os.path.join(comp_dir, f"{base_name}_encode.txt")
    dec_log = os.path.join(comp_dir, f"{base_name}_decode.txt")
    vstats_path = os.path.join(comp_dir, f"{base_name}_vstats.log")

    try:
        encode_cmd, encoder_name = build_encode_cmd(
            input_path=input_path,
            width=width,
            height=height,
            frames=frames,
            qp=qp,
            codec=codec,
            bitstream_path=bitstream_path,
            vstats_path=vstats_path,
            backend=backend,
            hw_device=hw_device,
        )
    except Exception as e:
        return {
            "codec": codec,
            "scene": scene_name,
            "qp": qp,
            "status": "config_fail",
            "error": str(e),
        }

    t0 = time.time()
    enc_rc = _run_command(encode_cmd, enc_log)
    enc_sec = round(time.time() - t0, 1)
    if enc_rc != 0:
        return {
            "codec": codec,
            "scene": scene_name,
            "qp": qp,
            "status": "enc_fail",
            "backend": backend,
            "encoder": encoder_name,
            "return_code": enc_rc,
            "enc_sec": enc_sec,
            "cmd": " ".join(shlex.quote(x) for x in encode_cmd),
            "log_path": enc_log,
            "log_tail": _tail_log(enc_log),
        }

    t1 = time.time()
    decode_cmd = build_decode_cmd(bitstream_path, recon_path, width, height, frames)
    dec_rc = _run_command(decode_cmd, dec_log)
    dec_sec = round(time.time() - t1, 1)
    if dec_rc != 0:
        return {
            "codec": codec,
            "scene": scene_name,
            "qp": qp,
            "status": "dec_fail",
            "backend": backend,
            "encoder": encoder_name,
            "return_code": dec_rc,
            "enc_sec": enc_sec,
            "dec_sec": dec_sec,
            "cmd": " ".join(shlex.quote(x) for x in decode_cmd),
            "log_path": dec_log,
            "log_tail": _tail_log(dec_log),
        }
    # After successful encode/decode, extract per-frame QP information
    qp_per_frame = parse_vstats_qps(vstats_path)
    qp_from_log = parse_encoder_log_qps(enc_log)

    qp_json_path = os.path.join(comp_dir, f"{base_name}_qp_per_frame.json")
    qp_log_json_path = os.path.join(comp_dir, f"{base_name}_qp_from_log.json")

    if qp_per_frame:
        save_qp_json(qp_json_path, {"source": "vstats", "frames": qp_per_frame})
    else:
        # If ffprobe didn't return QP info, but encoder log has numbers, save them as fallback
        if qp_from_log:
            frames = [{"index": i, "qp": q} for i, q in enumerate(qp_from_log)]
            save_qp_json(qp_json_path, {"source": "encoder_log_fallback", "frames": frames})

    if qp_from_log:
        save_qp_json(qp_log_json_path, {"source": "encoder_log", "qps": qp_from_log})

    return {
        "codec": codec,
        "scene": scene_name,
        "qp": qp,
        "status": "ok",
        "backend": backend,
        "encoder": encoder_name,
        "comp_dir": comp_dir,
        "bitstream": bitstream_path,
        "recon": recon_path,
        "enc_sec": enc_sec,
        "dec_sec": dec_sec,
        "qp_per_frame_json": qp_json_path if os.path.exists(qp_json_path) else None,
        "qp_from_log_json": qp_log_json_path if os.path.exists(qp_log_json_path) else None,
    }


def run_all(
    input_dir,
    output_base,
    qp_values,
    codecs_enabled,
    backend,
    dry_run=False,
    num_workers=N_WORKERS,
    hw_device=None,
    resume=True,
):
    yuv_files = sorted(glob.glob(os.path.join(input_dir, "*.yuv")))
    pattern = re.compile(r"(.+)_(\d+)x(\d+)_(\d+)\.yuv")

    valid_files = []
    for f in yuv_files:
        m = pattern.match(os.path.basename(f))
        if m:
            valid_files.append((f, *m.groups()))
        else:
            print(f"[Skip] 파일명 형식 오류: {os.path.basename(f)}")

    if not valid_files:
        print(f"[Error] {input_dir} 에서 유효한 .yuv 파일 없음 (형식: name_WxH_F.yuv)")
        return

    active_codecs = [c for c, en in codecs_enabled.items() if en]
    if not active_codecs:
        print("[Error] 활성화된 코덱이 없습니다.")
        return

    all_jobs = []
    skipped_existing = []
    for input_path, scene_name, width, height, frames in valid_files:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        width_i = int(width)
        height_i = int(height)
        frames_i = int(frames)
        for qp in qp_values:
            for codec in active_codecs:
                if resume and is_completed_job(
                    output_base=output_base,
                    codec=codec,
                    scene_name=scene_name,
                    qp=qp,
                    base_name=base_name,
                    width=width_i,
                    height=height_i,
                    frames=frames_i,
                ):
                    skipped_existing.append(
                        {
                            "codec": codec,
                            "scene": scene_name,
                            "qp": qp,
                            "status": "skipped_existing",
                        }
                    )
                    continue
                all_jobs.append(
                    {
                        "input_path": input_path,
                        "scene_name": scene_name,
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "codec": codec,
                        "qp": qp,
                        "output_base": output_base,
                        "base_name": base_name,
                        "backend": backend,
                        "hw_device": hw_device,
                    }
                )

    total = len(all_jobs)
    os.makedirs(output_base, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FFmpeg RA 압축 스크립트 시작")
    print(f"  파일 수      : {len(valid_files)}")
    print(f"  활성 코덱    : {active_codecs}")
    print(f"  백엔드       : {backend}")
    if hw_device:
        print(f"  HW 디바이스  : {hw_device}")
    print(f"  QP 목록      : {qp_values}")
    print(f"  총 작업 수   : {total}")
    print(f"  기존 완료 skip: {len(skipped_existing)}")
    print(f"  병렬 워커 수 : {num_workers}")
    print(f"  GOP / B      : {GOP_SIZE} / {B_FRAMES}")
    print(f"  출력 경로    : {output_base}")
    if dry_run:
        print("  [DRY-RUN 모드]")
    print("=" * 70 + "\n")

    if dry_run:
        for j in skipped_existing:
            print(f"(dry-run:skip) {j['codec']} | {j['scene']} | QP {j['qp']}")
        for j in all_jobs:
            print(f"(dry-run) {j['codec']} | {j['scene_name']} | QP {j['qp']}")
        return

    job_log = list(skipped_existing)
    ok_count = len(skipped_existing)
    fail_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_job, j): j for j in all_jobs}

        with tqdm(total=total, unit="job", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                job_def = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "codec": job_def["codec"],
                        "scene": job_def["scene_name"],
                        "qp": job_def["qp"],
                        "status": "exception",
                        "error": str(e),
                    }

                label = f"{result['codec']} | {result['scene']} | QP {result['qp']}"
                status = result.get("status", "?")

                if status == "ok":
                    ok_count += 1
                    tqdm.write(
                        f"  ✓ {label}  encoder={result.get('encoder')}  enc={result.get('enc_sec')}s  dec={result.get('dec_sec')}s"
                    )
                else:
                    fail_count += 1
                    lines = [f"  ✗ {label}  [{status}]"]
                    if "encoder" in result:
                        lines.append(f"    encoder : {result['encoder']}")
                    if "return_code" in result:
                        lines.append(f"    return_code : {result['return_code']}")
                    if "error" in result:
                        lines.append(f"    error : {result['error']}")
                    if result.get("cmd"):
                        lines.append(f"    cmd   : {result['cmd']}")
                    if result.get("log_tail"):
                        lines.append(f"    log   : {result.get('log_path', '')}")
                        for ln in result["log_tail"].splitlines():
                            if ln.strip():
                                lines.append(f"    > {ln.rstrip()}")
                    tqdm.write("\n".join(lines))

                job_log.append(result)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                pbar.update(1)

    elapsed = timedelta(seconds=int(time.time() - start_time))

    print("\n" + "=" * 70)
    print(f"  모든 작업 완료 (총 소요: {elapsed})")
    print(f"  성공/스킵: {ok_count}  실패: {fail_count}  총: {len(job_log)}")
    print("\n  [코덱별 통계]")
    for codec in active_codecs:
        entries = [j for j in job_log if j["codec"] == codec and j["status"] == "ok"]
        if entries:
            avg_enc = sum(j.get("enc_sec", 0) for j in entries) / len(entries)
            qps = [j["qp"] for j in entries]
            print(
                f"    {codec}: {len(entries)} jobs  "
                f"QP min={min(qps)} max={max(qps)} mean={sum(qps)/len(qps):.1f}  "
                f"평균 인코딩 {avg_enc:.1f}s"
            )

    log_path = os.path.join(output_base, "job_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(job_log, f, ensure_ascii=False, indent=2)

    print(f"\n  작업 로그 저장: {log_path}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FFmpeg 기반 RA 압축 스크립트 (AVC / HEVC / AV1 / VP9)"
    )
    parser.add_argument("--input", default=INPUT_DIR, help="YUV 파일 폴더 경로")
    parser.add_argument("--output", default=OUTPUT_BASE_DIR, help="결과 저장 폴더")
    parser.add_argument(
        "--qp-values",
        type=str,
        default="50,48,46,44,42,40,38,36,34,32,30",
        help="쉼표로 구분된 QP 리스트",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"병렬 워커 수 (기본: {N_WORKERS})",
    )
    parser.add_argument(
        "--codec",
        nargs="+",
        choices=["AVC", "HEVC", "AV1", "VP9"],
        default=None,
        help="실행할 코덱 선택 (미지정 시 전부 실행)",
    )
    parser.add_argument(
        "--backend",
        choices=["software", "nvenc", "qsv", "vaapi"],
        default="software",
        help="FFmpeg 인코더 백엔드",
    )
    parser.add_argument(
        "--hw-device",
        default=None,
        help="하드웨어 디바이스 경로. VAAPI 사용 시 예: /dev/dri/renderD128",
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 인코딩 없이 작업 목록만 출력")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="이미 완료된 작업은 건너뛰고 이어서 실행",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    qp_values = sorted(
        [int(x.strip()) for x in args.qp_values.split(",") if x.strip()],
        reverse=True,
    )
    for qp in qp_values:
        if qp < 20 or qp > 50:
            raise ValueError(f"QP must be in [20, 50] for this script: {qp}")

    codecs_enabled = {}
    for c, en in CODECS_ENABLED.items():
        codec_ok = (args.codec is None) or (c in args.codec)
        codecs_enabled[c] = en and codec_ok

    run_all(
        input_dir=args.input,
        output_base=args.output,
        qp_values=qp_values,
        codecs_enabled=codecs_enabled,
        backend=args.backend,
        dry_run=args.dry_run,
        num_workers=args.workers,
        hw_device=args.hw_device,
        resume=args.resume,
    )

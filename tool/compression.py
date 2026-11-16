import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
try:
    from tqdm import tqdm
except ImportError:
    import subprocess as _sp
    _sp.run(["pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
INPUT_DIR = r"./raw_data"
OUTPUT_BASE_DIR = r"./testset"

# Reference HEVC QP values; AV1/VP9 quality values are mapped below.

QP_VALUES = [27, 42]
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

# Reference HEVC QP to per-codec quality mapping
# AVC  : QP (same numeric value)
# HEVC : QP (reference)
# AV1  : SVT-AV1 CRF (0~63)
# VP9  : libvpx-vp9 CRF (0~63)

QUALITY_MAP = {
    27: {"AVC": 27, "HEVC": 27, "AV1": 32, "VP9": 33},
    42: {"AVC": 42, "HEVC": 42, "AV1": 50, "VP9": 52},
}

def get_quality(hevc_qp, codec):
    """Map a reference HEVC QP to the codec-specific quality value."""

    if hevc_qp not in QUALITY_MAP:
        raise ValueError(
            f"QP {hevc_qp} is not in QUALITY_MAP. "
            f"Defined keys: {list(QUALITY_MAP.keys())}"
        )

    return QUALITY_MAP[hevc_qp][codec]

def expected_yuv420p_size(width, height, frames):

    return width * height * 3 // 2 * frames

def _make_even_dimension(value):

    return int(value) if int(value) % 2 == 0 else int(value) - 1

def _make_even_size(width, height):

    return _make_even_dimension(width), _make_even_dimension(height)

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

def make_output_dirs(output_base, codec, scene_name, qp, work_base=None):

    frame_dir = os.path.join(output_base, codec, f"QP{qp}", scene_name)
    temp_dir = os.path.join(work_base or os.path.join(INPUT_DIR, "compressed"), codec, f"QP{qp}", scene_name)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    return frame_dir, temp_dir

def is_completed_job(output_base, codec, scene_name, qp, base_name, width, height, frames, work_base=None):

    frame_dir, _ = make_output_dirs(output_base, codec, scene_name, qp, work_base)
    frame_files = glob.glob(os.path.join(frame_dir, "*.png"))

    return len(frame_files) >= frames

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

def build_ra_video_options(codec, encoder, backend, hevc_qp, gop_size, b_frames):
    """
    hevc_qp: reference HEVC QP; actual codec settings come from QUALITY_MAP.
    AVC/HEVC : QP (fixed quantizer)
    AV1       : SVT-AV1 CRF (svtav1-params crf=)
    VP9       : libvpx-vp9 CRF + -b:v 0 (constrained quality mode)
    """
    quality = get_quality(hevc_qp, codec)

    fixed_gop_opts = [
        "-g", str(gop_size),
        "-keyint_min", str(gop_size),
        "-sc_threshold", "0",
    ]

    if backend == "software":
        if codec == "AVC":
            return fixed_gop_opts + [
                "-bf", str(b_frames),
                "-qp", str(quality),
                "-x264-params",
                (
                    f"keyint={gop_size}:min-keyint={gop_size}:"
                    f"scenecut=0:open-gop=0:bframes={b_frames}"
                ),
            ]

        if codec == "HEVC":
            return fixed_gop_opts + [
                "-bf", str(b_frames),
                "-qp", str(quality),
                "-x265-params",
                (
                    f"keyint={gop_size}:min-keyint={gop_size}:"
                    f"scenecut=0:open-gop=0:bframes={b_frames}"
                ),
            ]

        if codec == "AV1":
            return [
                "-crf", str(quality),
                "-preset", "6",
                "-svtav1-params",
                (
                    f"keyint={gop_size}:min-keyint={gop_size}:"
                    f"scd=0:irefresh-type=2:hierarchical-levels=4"
                ),
            ]

        if codec == "VP9":
            return [
                "-g", str(gop_size),
                "-keyint_min", str(gop_size),
                "-skip_threshold", "0",
                "-crf", str(quality),
                "-b:v", "0",
                "-row-mt", "1",
                "-tile-columns", "2",
                "-aq-mode", "0",
                "-lag-in-frames", "25",
                "-auto-alt-ref", "1",
            ]

    if backend == "nvenc":
        opts = fixed_gop_opts
        if codec in {"AVC", "HEVC"}:
            opts += ["-bf", str(b_frames), "-forced-idr", "1"]
        opts += ["-rc", "constqp", "-qp", str(quality), "-preset", "p5"]

        return opts

    if backend == "qsv":
        opts = fixed_gop_opts
        if codec in {"AVC", "HEVC"}:
            opts += ["-bf", str(b_frames)]
        opts += ["-global_quality", str(quality), "-look_ahead", "0"]

        return opts

    if backend == "vaapi":
        opts = fixed_gop_opts

        if codec in {"AVC", "HEVC"}:
            opts += ["-bf", str(b_frames)]

        opts += ["-qp", str(quality)]
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
    hevc_qp,
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
    cmd += build_ra_video_options(codec, encoder, backend, hevc_qp, GOP_SIZE, B_FRAMES)
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

def build_image_sequence_to_yuv_cmd(input_list_path, yuv_path, width, height, frames):

    return [
        FFMPEG_PATH,
        "-y",
        "-r", str(FPS),
        "-f", "concat",
        "-safe", "0",
        "-i", input_list_path,
        "-frames:v", str(frames),
        "-vf", f"scale={width}:{height}:flags=bicubic,format=yuv420p",
        "-f", "rawvideo",
        yuv_path,
    ]

def build_decode_frames_cmd(bitstream_path, frame_pattern, frames):

    return [
        FFMPEG_PATH,
        "-y",
        "-i", bitstream_path,
        "-frames:v", str(frames),
        "-pix_fmt", "rgb24",
        "-start_number", "1",
        frame_pattern,
    ]

def parse_vstats_qps(vstats_path):

    frames = []

    if not os.path.exists(vstats_path):
        return frames

    frame_re = re.compile(r"\bframe=\s*(\d+)")
    qp_re = re.compile(r"\bq=\s*([\\-\d.]+)")
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

    try:
        width = int(job["width"])
        height = int(job["height"])
        frames = int(job["frames"])

    except (TypeError, ValueError) as e:
        return {
            "codec": job.get("codec"),
            "scene": scene_name,
            "hevc_qp": job.get("hevc_qp"),
            "quality": job.get("quality"),
            "status": "config_fail",
            "error": f"invalid image metadata: width={job.get('width')} "
                     f"height={job.get('height')} frames={job.get('frames')} ({e})",
        }

    codec = job["codec"]
    hevc_qp = int(job["hevc_qp"])          # Reference HEVC QP used in output paths
    quality = int(job["quality"])           # Actual codec-specific quality value
    output_base = job["output_base"]
    base_name = job["base_name"]
    backend = job["backend"]
    hw_device = job.get("hw_device")
    input_kind = job.get("input_kind", "raw")

    frame_dir, temp_dir = make_output_dirs(output_base, codec, scene_name, hevc_qp, job["work_base"])
    temp_yuv_path = os.path.join(temp_dir, f"{base_name}_input.yuv")
    bitstream_path = os.path.join(temp_dir, f"{base_name}{BITSTREAM_EXT[codec]}")
    enc_log = os.path.join(temp_dir, f"{base_name}_encode.txt")
    dec_log = os.path.join(temp_dir, f"{base_name}_decode.txt")
    yuv_log = os.path.join(temp_dir, f"{base_name}_to_yuv.txt")
    image_list_path = os.path.join(temp_dir, f"{base_name}_images.txt")
    vstats_path = os.path.join(temp_dir, f"{base_name}_vstats.log")

    if input_kind == "image_sequence":
        expected_yuv_size = expected_yuv420p_size(width, height, frames)

        if (
            not os.path.exists(temp_yuv_path)
            or os.path.getsize(temp_yuv_path) != expected_yuv_size
        ):

            with open(image_list_path, "w", encoding="utf-8") as f:
                for image_path in job["image_files"]:
                    safe_path = image_path.replace("\\\\", "/").replace("'", r"'\\''")
                    f.write(f"file '{safe_path}'\n")

            yuv_cmd = build_image_sequence_to_yuv_cmd(
                input_list_path=image_list_path,
                yuv_path=temp_yuv_path,
                width=width,
                height=height,
                frames=frames,
            )

            yuv_rc = _run_command(yuv_cmd, yuv_log)

            if yuv_rc != 0:
                return {
                    "codec": codec,
                    "scene": scene_name,
                    "hevc_qp": hevc_qp,
                    "quality": quality,
                    "status": "yuv_fail",
                    "return_code": yuv_rc,
                    "cmd": " ".join(shlex.quote(x) for x in yuv_cmd),
                    "log_path": yuv_log,
                    "log_tail": _tail_log(yuv_log),
                }
        input_path = temp_yuv_path

    try:

        encode_cmd, encoder_name = build_encode_cmd(
            input_path=input_path,
            width=width,
            height=height,
            frames=frames,
            hevc_qp=hevc_qp,
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
            "hevc_qp": hevc_qp,
            "quality": quality,
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
            "hevc_qp": hevc_qp,
            "quality": quality,
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

    frame_pattern = os.path.join(frame_dir, "%08d.png")
    decode_cmd = build_decode_frames_cmd(bitstream_path, frame_pattern, frames)
    dec_rc = _run_command(decode_cmd, dec_log)
    dec_sec = round(time.time() - t1, 1)

    if dec_rc != 0:
        return {
            "codec": codec,
            "scene": scene_name,
            "hevc_qp": hevc_qp,
            "quality": quality,
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

    qp_per_frame = parse_vstats_qps(vstats_path)
    qp_from_log = parse_encoder_log_qps(enc_log)
    qp_json_path = os.path.join(temp_dir, f"{base_name}_qp_per_frame.json")
    qp_log_json_path = os.path.join(temp_dir, f"{base_name}_qp_from_log.json")

    if qp_per_frame:
        save_qp_json(qp_json_path, {"source": "vstats", "frames": qp_per_frame})

    else:

        if qp_from_log:
            fallback_frames = [{"index": i, "qp": q} for i, q in enumerate(qp_from_log)]

            save_qp_json(
                qp_json_path,
                {"source": "encoder_log_fallback", "frames": fallback_frames},
            )

    if qp_from_log:
        save_qp_json(qp_log_json_path, {"source": "encoder_log", "qps": qp_from_log})

    return {
        "codec": codec,
        "scene": scene_name,
        "hevc_qp": hevc_qp,
        "quality": quality,
        "status": "ok",
        "backend": backend,
        "encoder": encoder_name,
        "frame_dir": frame_dir,
        "temp_dir": temp_dir,
        "bitstream": bitstream_path,
        "enc_sec": enc_sec,
        "dec_sec": dec_sec,
        "qp_per_frame_json": qp_json_path if os.path.exists(qp_json_path) else None,
        "qp_from_log_json": qp_log_json_path if os.path.exists(qp_log_json_path) else None,
    }

def _ffprobe_path():
    if os.path.basename(FFMPEG_PATH).lower().startswith("ffmpeg"):
        return os.path.join(os.path.dirname(FFMPEG_PATH), "ffprobe")

    return "ffprobe"

def _probe_image_size(path):

    ffprobe_cmd = [
        _ffprobe_path(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        path,
    ]

    try:
        out = subprocess.check_output(ffprobe_cmd, text=True, stderr=subprocess.DEVNULL).strip()
        width, height = out.split("x", 1)
        return int(width), int(height)

    except Exception:
        pass

    ffmpeg_cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-i", path,
        "-frames:v", "1",
        "-f", "null",
        "-",
    ]

    try:
        proc = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )

        match = re.search(r"\b(\d{2,5})x(\d{2,5})\b", proc.stdout)

        if match:
            return int(match.group(1)), int(match.group(2))

    except Exception:
        pass

    return None, None

def _discover_raw_yuv_files(input_dir):
    yuv_files = sorted(glob.glob(os.path.join(input_dir, "*.yuv")))
    pattern = re.compile(r"(.+)_(\d+)x(\d+)_(\d+)\.yuv")

    valid_files = []

    for f in yuv_files:
        m = pattern.match(os.path.basename(f))

        if not m:
            print(f"[Skip] invalid YUV filename: {os.path.basename(f)}")
            continue

        scene_name, width, height, frames = m.groups()

        valid_files.append(

            {
                "input_path": f,
                "scene_name": scene_name,
                "width": int(width),
                "height": int(height),
                "frames": int(frames),
                "base_name": os.path.splitext(os.path.basename(f))[0],
                "input_kind": "raw",
            }
        )

    return valid_files

def _split_sequence_name(filename):

    stem, ext = os.path.splitext(filename)

    match = re.match(r"^(.*?)(\d+)$", stem)

    if not match:
        return None

    prefix, number = match.groups()

    return prefix, number, ext

def _discover_image_sequences(input_dir, excluded_dirs=()):

    exts = (".jpg", ".jpeg", ".png")

    sequences = []

    excluded = [os.path.realpath(p) for p in excluded_dirs]
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not any(
            os.path.realpath(os.path.join(root, d)) == p for p in excluded)]
        if os.path.realpath(root) in excluded:
            dirs[:] = []
            continue

        images = sorted(f for f in files if os.path.splitext(f)[1].lower() in exts)

        if not images:
            continue

        first_parts = _split_sequence_name(images[0])

        if not first_parts:
            continue

        prefix, first_number, first_ext = first_parts
        digit_count = len(first_number)

        numeric = [
            f for f in images
            if (parts := _split_sequence_name(f))
            and parts[0] == prefix
            and len(parts[1]) == digit_count
            and parts[2].lower() == first_ext.lower()
        ]

        if not numeric:
            continue

        numeric = sorted(numeric, key=lambda f: int(_split_sequence_name(f)[1]))
        rel_scene = os.path.relpath(root, input_dir)
        first_path = os.path.join(root, numeric[0])
        width, height = _probe_image_size(first_path)

        if width is None or height is None:
            print(f"[Skip] cannot read image size: {first_path}")
            continue

        width, height = _make_even_size(width, height)
        start_number = int(first_number)
        sequences.append(

            {
                "input_path": os.path.join(root, f"{prefix}%0{digit_count}d{first_ext}"),
                "scene_name": rel_scene,
                "width": width,
                "height": height,
                "frames": len(numeric),
                "base_name": os.path.basename(root),
                "input_kind": "image_sequence",
                "start_number": start_number,
                "image_files": [os.path.abspath(os.path.join(root, f)) for f in numeric],
            }
        )

    return sequences

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

    work_base = os.path.join(input_dir, "compressed")
    valid_files = _discover_raw_yuv_files(input_dir)

    if not valid_files:
        valid_files = _discover_image_sequences(input_dir, (work_base, output_base))

    unused_yuv_files = []
    pattern = re.compile(r"a^")

    for f in unused_yuv_files:
        m = pattern.match(os.path.basename(f))

        if m:
            valid_files.append((f, *m.groups()))

        else:
            print(f"[Skip] Invalid filename: {os.path.basename(f)}")

    if not valid_files:
        print(f"[Error] No valid YUV files or image sequences found in {input_dir}")
        return

    active_codecs = [c for c, en in codecs_enabled.items() if en]

    if not active_codecs:
        print("[Error] No codecs are enabled.")
        return

    all_jobs = []

    skipped_existing = []

    for item in valid_files:
        input_path = item["input_path"]
        scene_name = item["scene_name"]
        base_name = item["base_name"]
        width_i = int(item["width"]) if item["width"] else None
        height_i = int(item["height"]) if item["height"] else None
        frames_i = int(item["frames"])

        for hevc_qp in qp_values:
            for codec in active_codecs:
                quality = get_quality(hevc_qp, codec)

                if resume and is_completed_job(
                    output_base=output_base,
                    codec=codec,
                    scene_name=scene_name,
                    qp=hevc_qp,
                    base_name=base_name,
                    width=width_i,
                    height=height_i,
                    frames=frames_i,
                    work_base=work_base,
                ):

                    skipped_existing.append(
                        {
                            "codec": codec,
                            "scene": scene_name,
                            "hevc_qp": hevc_qp,
                            "quality": quality,
                            "status": "skipped_existing",
                        }
                    )

                    continue

                all_jobs.append(
                    {
                        "input_path": input_path,
                        "scene_name": scene_name,
                        "width": width_i,
                        "height": height_i,
                        "frames": frames_i,
                        "codec": codec,
                        "hevc_qp": hevc_qp,
                        "quality": quality,
                        "output_base": output_base,
                        "work_base": work_base,
                        "base_name": base_name,
                        "backend": backend,
                        "hw_device": hw_device,
                        "input_kind": item["input_kind"],
                        "start_number": item.get("start_number", 0),
                        "image_files": item.get("image_files"),
                    }
                )

    total = len(all_jobs)
    os.makedirs(output_base, exist_ok=True)

    # Format QUALITY_MAP for display
    qmap_lines = []

    for hqp in qp_values:
        vals = QUALITY_MAP[hqp]
        qmap_lines.append(
            f"  HEVC QP {hqp:2d} -> "
            + "  ".join(f"{c}:{vals[c]:2d}" for c in active_codecs)
        )

    print("\n" + "=" * 70)
    print("  Starting FFmpeg compression")
    print(f"  Sequence count        : {len(valid_files)}")
    print(f"  Enabled codecs      : {active_codecs}")
    print(f"  Backend         : {backend}")

    if hw_device:
        print(f"  HW device    : {hw_device}")

    print(f"  Reference HEVC QPs   : {qp_values}")
    print("  Quality mapping   :")

    for line in qmap_lines:
        print(line)

    print(f"  Total jobs     : {total}")
    print(f"  Already completed : {len(skipped_existing)}")
    print(f"  Workers   : {num_workers}")
    print(f"  GOP / B        : {GOP_SIZE} / {B_FRAMES}")
    print(f"  Output path      : {output_base}")

    if dry_run:
        print("  [DRY-RUN mode]")
    print("=" * 70 + "\n")

    if dry_run:
        for j in skipped_existing:
            print(
                f"(dry-run:skip) {j['codec']} | {j['scene']} "
                f"| HEVC QP {j['hevc_qp']} (quality={j['quality']})"
            )

        for j in all_jobs:
            print(
                f"(dry-run) {j['codec']} | {j['scene_name']} "
                f"| HEVC QP {j['hevc_qp']} (quality={j['quality']})"
            )

        return

    # Preserve original RGB files and relative paths in the GT directory.
    # Raw YUV inputs do not contain original RGB files to copy.
    for item in valid_files:
        if item["input_kind"] != "image_sequence":
            continue
        target_dir = os.path.join(output_base, "GT", item["scene_name"])
        os.makedirs(target_dir, exist_ok=True)
        for image_path in item["image_files"]:
            target = os.path.join(target_dir, os.path.basename(image_path))
            if os.path.exists(target):
                with open(image_path, "rb") as src, open(target, "rb") as dst:
                    if src.read() != dst.read():
                        raise FileExistsError(f"GT file differs from source: {target}")
            else:
                shutil.copy2(image_path, target)

    job_log = list(skipped_existing)
    ok_count = len(skipped_existing)
    fail_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
                        "hevc_qp": job_def["hevc_qp"],
                        "quality": job_def["quality"],
                        "status": "exception",
                        "error": str(e),
                    }

                label = (
                    f"{result['codec']} | {result['scene']} "
                    f"| HEVC QP {result['hevc_qp']} (quality={result['quality']})"
                )

                status = result.get("status", "?")

                if status == "ok":
                    ok_count += 1
                    tqdm.write(
                        f"  [OK] {label}  encoder={result.get('encoder')}"
                        f"  enc={result.get('enc_sec')}s  dec={result.get('dec_sec')}s"
                    )

                else:
                    fail_count += 1
                    lines = [f"  [FAIL] {label}  [{status}]"]

                    if "encoder" in result:
                        lines.append(f"    encoder     : {result['encoder']}")

                    if "return_code" in result:
                        lines.append(f"    return_code : {result['return_code']}")

                    if "error" in result:
                        lines.append(f"    error       : {result['error']}")

                    if result.get("cmd"):
                        lines.append(f"    cmd         : {result['cmd']}")

                    if result.get("log_tail"):
                        lines.append(f"    log         : {result.get('log_path', '')}")
                        for ln in result["log_tail"].splitlines():
                            if ln.strip():
                                lines.append(f"    > {ln.rstrip()}")
                    tqdm.write("\n".join(lines))

                job_log.append(result)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                pbar.update(1)

    elapsed = timedelta(seconds=int(time.time() - start_time))

    print("\n" + "=" * 70)
    print(f"  All jobs finished (elapsed: {elapsed})")
    print(f"  Succeeded/skipped: {ok_count}  Failed: {fail_count}  Total: {len(job_log)}")
    print("\n  [Per-codec summary]")

    for codec in active_codecs:
        entries = [j for j in job_log if j["codec"] == codec and j["status"] == "ok"]
        if entries:
            avg_enc = sum(j.get("enc_sec", 0) for j in entries) / len(entries)
            qps = [j["hevc_qp"] for j in entries]
            print(
                f"    {codec}: {len(entries)} jobs  "
                f"HEVC QP min={min(qps)} max={max(qps)}  "
                f"Mean encoding time {avg_enc:.1f}s"
            )

    os.makedirs(work_base, exist_ok=True)
    log_path = os.path.join(work_base, "job_log.json")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(job_log, f, ensure_ascii=False, indent=2)

    print(f"\n  Job log saved: {log_path}")
    print("=" * 70 + "\n")

def parse_args():

    parser = argparse.ArgumentParser(
        description="Compress raw YUV or RGB sequences and export PNGs (AVC / HEVC / AV1 / VP9)"
    )
    parser.add_argument("--input", default=INPUT_DIR, help="Input root containing dataset/scene RGB folders or raw YUV files")
    parser.add_argument("--output", default=OUTPUT_BASE_DIR, help="Final PNG output root (default: ./testset)")
    parser.add_argument(
        "--qp-values",
        type=str,
        default=",".join(str(q) for q in QP_VALUES),
        help="Comma-separated reference HEVC QPs defined in QUALITY_MAP",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"Workers (default: {N_WORKERS})",
    )
    parser.add_argument(
        "--codec",
        nargs="+",
        choices=["AVC", "HEVC", "AV1", "VP9"],
        default=None,
        help="Select from codecs enabled in CODECS_ENABLED",
    )
    parser.add_argument(
        "--backend",
        choices=["software", "nvenc", "qsv", "vaapi"],
        default="software",
        help="FFmpeg encoder backend",
    )
    parser.add_argument(
        "--hw-device",
        default=None,
        help="Hardware device path, e.g. /dev/dri/renderD128 for VAAPI",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without encoding or copying GT")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed jobs and resume processing",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    qp_values = sorted(
        [int(x.strip()) for x in args.qp_values.split(",") if x.strip()],
        reverse=True,
    )

    # Reject QPs missing from QUALITY_MAP
    for qp in qp_values:
        if qp not in QUALITY_MAP:
            raise ValueError(
                f"QP {qp} is not defined in QUALITY_MAP. "
                f"Available: {sorted(QUALITY_MAP.keys())}"
            )

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

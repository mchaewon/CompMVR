import os
import os.path as op
import glob
import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
import pyiqa

from diffusion.CompMVR import CompMVR_test
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from cpe.cpe import CPE


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--compmvr_path", type=str, required=True)
    parser.add_argument("--comp_path", type=str, required=True)

    parser.add_argument("--testset_root", type=str, required=True)
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="./results/inference")

    # Dataset filter: Free / Replica / CO3Dv2 / Re10k
    parser.add_argument("--datasets", type=str, required=True)

    # Optional codec / qp filter
    parser.add_argument("--codecs", type=str, default=None)  # e.g., AV1,HEVC
    parser.add_argument("--qps", type=str, default=None)     # e.g., QP27,QP42

    parser.add_argument("--use_intra", action="store_true", default=False)
    parser.add_argument("--use_inter", action="store_true", default=False)

    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--timestep", type=int, default=499)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["fp16", "fp32"])
    parser.add_argument("--merge_and_unload_lora", action="store_true", default=False)

    parser.add_argument("--use_r_2D", action="store_true", default=False)
    parser.add_argument("--use_r_3D", action="store_true", default=False)
    parser.add_argument("--use_spatial_r", action="store_true", default=False)
    parser.add_argument("--use_vae_r", action="store_true", default=False)

    parser.add_argument("--align_method", type=str, default="adain",
                        choices=["wavelet", "adain", "nofix"])
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    return parser.parse_args()


def validate_args(args):
    if not args.use_inter:
        args.num_views = 1


# =========================
# Basic utils
# =========================
def get_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(op.join(folder, ext)))
    return sorted(files)


def list_subdirs(path):
    if not op.isdir(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if op.isdir(op.join(path, d))
    ])


def parse_filter_list(s):
    if s is None or str(s).strip() == "":
        return None
    return set(x.strip() for x in s.split(","))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# Dataset structure parser
# =========================
def collect_dataset_scenes(testset_root, gt_root, dataset_name, target_codecs=None, target_qps=None):
    """
    Returns:
        list of dict:
        {
            codec, qp, dataset, scene,
            lq_imgs, gt_imgs
        }

    Supported structures:

    Free:
      LQ: testset/AV1/QP27/Free/grass/*.png
      GT: testset/GT/Free/grass/*.JPG

    Replica:
      LQ: testset/AV1/QP27/Replica/AV1_office3_QP27/frame/*.png
      GT: testset/GT/Replica/office3/results/*.jpg

    CO3Dv2:
      LQ: testset/AV1/QP27/CO3Dv2/434_61668_121209/*.png
          or deeper leaf dirs
      GT: testset/GT/CO3Dv2/434_61668_121209/*.jpg

    Re10k:
      LQ: testset/AV1/QP27/Re10k/49c758aa3c35ed86/*.png
      GT: testset/GT/Re10k/49c758aa3c35ed86/*.png
    """
    dataset_name = dataset_name.strip()
    scenes = []

    codec_dirs = list_subdirs(testset_root)
    codec_dirs = [c for c in codec_dirs if c != "GT"]

    for codec in codec_dirs:
        if target_codecs is not None and codec not in target_codecs:
            continue

        codec_root = op.join(testset_root, codec)
        for qp in list_subdirs(codec_root):
            if target_qps is not None and qp not in target_qps:
                continue

            ds_root = op.join(codec_root, qp, dataset_name)
            if not op.isdir(ds_root):
                continue

            if dataset_name == "Free":
                for scene in list_subdirs(ds_root):
                    lq_dir = op.join(ds_root, scene)
                    gt_dir = op.join(gt_root, "Free", scene)

                    lq_imgs = get_images(lq_dir)
                    gt_imgs = get_images(gt_dir)

                    if lq_imgs and gt_imgs:
                        scenes.append({
                            "codec": codec,
                            "qp": qp,
                            "dataset": dataset_name,
                            "scene": scene,
                            "lq_imgs": lq_imgs,
                            "gt_imgs": gt_imgs,
                        })

            elif dataset_name == "Replica":
                for lq_scene_name in list_subdirs(ds_root):
                    # Example: AV1_office3_QP27 -> office3
                    scene = lq_scene_name

                    prefix = codec + "_"
                    suffix = "_" + qp
                    if scene.startswith(prefix):
                        scene = scene[len(prefix):]
                    if scene.endswith(suffix):
                        scene = scene[:-len(suffix)]

                    lq_scene_root = op.join(ds_root, lq_scene_name)

                    # Prefer frame folder if exists, otherwise recursive images.
                    frame_dir = op.join(lq_scene_root, "frame")
                    if op.isdir(frame_dir):
                        lq_imgs = get_images(frame_dir)
                    else:
                        lq_imgs = []
                        for root, _, _ in os.walk(lq_scene_root):
                            lq_imgs.extend(get_images(root))
                        lq_imgs = sorted(lq_imgs)

                    gt_dir = op.join(gt_root, "Replica", scene, "results")
                    gt_imgs = get_images(gt_dir)

                    if lq_imgs and gt_imgs:
                        scenes.append({
                            "codec": codec,
                            "qp": qp,
                            "dataset": dataset_name,
                            "scene": scene,
                            "lq_imgs": lq_imgs,
                            "gt_imgs": gt_imgs,
                        })

            elif dataset_name == "CO3Dv2":
                for scene in list_subdirs(ds_root):
                    lq_scene_root = op.join(ds_root, scene)
                    gt_scene_root = op.join(gt_root, "CO3Dv2", scene)

                    lq_imgs = []
                    for root, _, _ in os.walk(lq_scene_root):
                        lq_imgs.extend(get_images(root))
                    lq_imgs = sorted(lq_imgs)

                    gt_imgs = []
                    for root, _, _ in os.walk(gt_scene_root):
                        gt_imgs.extend(get_images(root))
                    gt_imgs = sorted(gt_imgs)

                    if lq_imgs and gt_imgs:
                        scenes.append({
                            "codec": codec,
                            "qp": qp,
                            "dataset": dataset_name,
                            "scene": scene,
                            "lq_imgs": lq_imgs,
                            "gt_imgs": gt_imgs,
                        })

            elif dataset_name in ["Re10K", "Re10k", "re10k"]:
                canonical_name = "Re10k"

                for scene in list_subdirs(ds_root):
                    lq_scene_root = op.join(ds_root, scene)
                    gt_scene_root = op.join(gt_root, "Re10k", scene)

                    lq_imgs = []
                    for root, _, _ in os.walk(lq_scene_root):
                        lq_imgs.extend(get_images(root))
                    lq_imgs = sorted(lq_imgs)

                    gt_imgs = []
                    for root, _, _ in os.walk(gt_scene_root):
                        gt_imgs.extend(get_images(root))
                    gt_imgs = sorted(gt_imgs)

                    if lq_imgs and gt_imgs:
                        scenes.append({
                            "codec": codec,
                            "qp": qp,
                            "dataset": canonical_name,
                            "scene": scene,
                            "lq_imgs": lq_imgs,
                            "gt_imgs": gt_imgs,
                        })

            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return scenes


# =========================
# Image I/O
# =========================
def load_lq(path, device):
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]

    new_w = w - w % 8
    new_h = h - h % 8
    img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))

    t01 = torch.tensor(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    tm11 = t01 * 2.0 - 1.0

    return tm11, t01, img


def load_gt(path, device):
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]

    new_w = w - w % 8
    new_h = h - h % 8
    img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))

    t01 = torch.tensor(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    return t01


def resize_pred_to_gt(pred, gt):
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bicubic", align_corners=False)
    return pred.clamp(0, 1)


def postprocess(out_t, comp_img, align_method):
    """
    out_t: (1, 3, H, W), expected [-1, 1]
    """
    out_01 = (out_t[0].detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
    out_pil = transforms.ToPILImage()(out_01)

    if align_method == "adain":
        out_pil = adain_color_fix(target=out_pil, source=comp_img)
    elif align_method == "wavelet":
        out_pil = wavelet_color_fix(target=out_pil, source=comp_img)

    return out_pil


# =========================
# Inference
# =========================
def infer_multiview(model, comp, img_paths, num_views, align_method, device):
    """
    Chunk-based inference:
      frames 0~V-1 -> V outputs
      frames V~2V-1 -> V outputs
    """
    results = []

    for i in range(0, len(img_paths), num_views):
        chunk = img_paths[i:i + num_views]
        real_count = len(chunk)
        original_names = [op.basename(p) for p in chunk]

        while len(chunk) < num_views:
            chunk.append(chunk[-1])

        lqs, lq_arrs, comp_imgs = [], [], []

        for p in chunk:
            lq, lq_arr, comp_img = load_lq(p, device)
            lqs.append(lq)
            lq_arrs.append(lq_arr)
            comp_imgs.append(comp_img)

        lq_flat = torch.cat(lqs, dim=0)          # [V, 3, H, W], [-1,1]
        lq_arr_flat = torch.cat(lq_arrs, dim=0)  # [V, 3, H, W], [0,1]

        with torch.no_grad():
            emb = comp.get_visual_embedding(lq_arr_flat)
            model._current_real_views = real_count
            out_flat = model(lq_flat, emb)       # [V, 3, H, W], expected [-1,1]

        for v in range(real_count):
            out_v = out_flat[v].unsqueeze(0)
            out_pil = postprocess(out_v, comp_imgs[v], align_method)
            results.append((original_names[v], out_pil))

    return results


# =========================
# Metric
# =========================
def evaluate_and_save(
    results,
    gt_imgs,
    save_dir,
    codec,
    qp,
    dataset,
    scene,
    metrics,
    metric_keys,
    fr_keys,
    frame_csv,
    scene_csv,
    device,
):
    ensure_dir(save_dir)
    scores = {k: [] for k in metric_keys}

    n = min(len(results), len(gt_imgs))

    for i in tqdm(range(n), desc=f"{scene}", leave=False):
        name, out_pil = results[i]

        save_path = op.join(save_dir, name)
        out_pil.save(save_path)

        pred = transforms.ToTensor()(out_pil).unsqueeze(0).to(device)
        gt = load_gt(gt_imgs[i], device)
        pred_r = resize_pred_to_gt(pred, gt)

        vals = {}

        with torch.no_grad():
            for k in metric_keys:
                if k in fr_keys:
                    vals[k] = metrics[k](pred_r, gt).item()
                else:
                    vals[k] = metrics[k](pred_r).item()
                scores[k].append(vals[k])

        with open(frame_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [codec, qp, dataset, scene, name]
                + [vals.get(k, 0.0) for k in metric_keys]
            )

    scene_avg = [float(np.mean(scores[k])) if scores[k] else 0.0 for k in metric_keys]

    with open(scene_csv, "a", newline="") as f:
        csv.writer(f).writerow([codec, qp, dataset, scene] + scene_avg)

    return scene_avg


# =========================
# Main
# =========================
def main():
    args = parse_args()
    validate_args(args)

    device = args.device
    ensure_dir(args.save_root)

    target_codecs = parse_filter_list(args.codecs)
    target_qps = parse_filter_list(args.qps)

    metric_keys = [
        "psnr", "msssim", "lpips", "dists",
        "niqe", "musiq", "clipiqa"
    ]
    fr_keys = {"psnr", "msssim", "lpips", "dists"}

    metrics = {
        "psnr": pyiqa.create_metric("psnr", device=device),
        "msssim": pyiqa.create_metric("ms_ssim", device=device),
        "lpips": pyiqa.create_metric("lpips-vgg", device=device),
        "dists": pyiqa.create_metric("dists", device=device),
        "niqe": pyiqa.create_metric("niqe", device=device),
        "musiq": pyiqa.create_metric("musiq", device=device),
        "clipiqa": pyiqa.create_metric("clipiqa+", device=device),
    }

    model = CompMVR_test(args).to(device).eval()

    comp = CPE(3, 3, [64, 128, 256, 512], 4, "BR")
    comp.load_state_dict(torch.load(args.comp_path, map_location=device))
    comp.to(device).eval()

    chunk_size = args.num_views if args.use_inter else 1

    print(
        f"[Mode] dataset={args.datasets}, use_intra={args.use_intra}, "
        f"use_inter={args.use_inter}, num_views={chunk_size}, "
        f"use_spatial_r={args.use_spatial_r}, use_vae_r={args.use_vae_r}"
    )

    scenes = collect_dataset_scenes(
        testset_root=args.testset_root,
        gt_root=args.gt_root,
        dataset_name=args.datasets,
        target_codecs=target_codecs,
        target_qps=target_qps,
    )

    if len(scenes) == 0:
        print(f"[ERROR] No valid scenes found for dataset={args.datasets}")
        return

    dataset_scores = {}

    for item in scenes:
        codec = item["codec"]
        qp = item["qp"]
        dataset = item["dataset"]
        scene = item["scene"]
        lq_imgs = item["lq_imgs"]
        gt_imgs = item["gt_imgs"]

        n = min(len(lq_imgs), len(gt_imgs))
        lq_imgs = lq_imgs[:n]
        gt_imgs = gt_imgs[:n]

        print(f"\n[{codec}][{qp}][{dataset}][{scene}] {n} frames (chunk={chunk_size})")

        save_dir = op.join(args.save_root, "images", codec, qp, dataset, scene)

        frame_csv = op.join(args.save_root, f"{dataset}_frame.csv")
        scene_csv = op.join(args.save_root, f"{dataset}_scene.csv")
        overall_csv = op.join(args.save_root, f"{dataset}_overall.csv")

        if dataset not in dataset_scores:
            dataset_scores[dataset] = {"scores": [], "overall": overall_csv}

            with open(frame_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["codec", "qp", "dataset", "scene", "frame"] + metric_keys
                )

            with open(scene_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["codec", "qp", "dataset", "scene"] + metric_keys
                )

        results = infer_multiview(
            model=model,
            comp=comp,
            img_paths=lq_imgs,
            num_views=chunk_size,
            align_method=args.align_method,
            device=device,
        )

        scene_avg = evaluate_and_save(
            results=results,
            gt_imgs=gt_imgs,
            save_dir=save_dir,
            codec=codec,
            qp=qp,
            dataset=dataset,
            scene=scene,
            metrics=metrics,
            metric_keys=metric_keys,
            fr_keys=fr_keys,
            frame_csv=frame_csv,
            scene_csv=scene_csv,
            device=device,
        )

        dataset_scores[dataset]["scores"].append(scene_avg)

        print("  " + " | ".join(
            f"{k}={v:.4f}" for k, v in zip(metric_keys, scene_avg)
        ))

    for dataset, pack in dataset_scores.items():
        overall = np.mean(np.array(pack["scores"], dtype=np.float32), axis=0)

        with open(pack["overall"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metric_keys)
            writer.writerow([f"{v:.4f}" for v in overall])

        print(f"\n=== {dataset} Overall ===")
        for k, v in zip(metric_keys, overall):
            print(f"  {k}: {v:.4f}")

    print(f"\n=== DONE === {args.save_root}")


if __name__ == "__main__":
    main()
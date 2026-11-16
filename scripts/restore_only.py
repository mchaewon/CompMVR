"""
restore_only.py

Restore every frame in a single input folder with CompMVR.
No GT, no metrics -- restoration only.

Frames are sorted by filename (natural sort) and processed in consecutive
chunks of `num_views` when --use_inter is set, so the sorted order should
follow the view / capture order.

Run from the repository root (diffusion/CompMVR.py relies on the cwd):
    python scripts/restore_only.py --input_dir ... --output_dir ... [options]
"""

import os
import os.path as op
import re
import sys
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

REPO_ROOT = op.dirname(op.dirname(op.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion.CompMVR import CompMVR_test
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from cpe.cpe import CPE


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="CompMVR restoration only (no GT, no metrics)")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing the compressed frames of ONE sequence")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Restored frames are saved here with their original filenames")

    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--compmvr_path", type=str, required=True)
    parser.add_argument("--comp_path", type=str, required=True)

    parser.add_argument("--use_intra", action="store_true", default=False)
    parser.add_argument("--use_inter", action="store_true", default=False)
    parser.add_argument("--use_spatial_r", action="store_true", default=False)
    parser.add_argument("--num_views", type=int, default=1)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--timestep", type=int, default=499)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--merge_and_unload_lora", action="store_true", default=False)

    parser.add_argument("--align_method", type=str, default="adain",
                        choices=["wavelet", "adain", "nofix"])
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    args = parser.parse_args()
    if not args.use_inter:
        args.num_views = 1
    return args


# =========================
# Utils
# =========================
def natural_key(path):
    name = op.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(folder):
    if not op.isdir(folder):
        raise FileNotFoundError(f"input_dir not found: {folder}")
    files = [
        op.join(folder, f) for f in os.listdir(folder)
        if op.isfile(op.join(folder, f)) and f.lower().endswith(IMG_EXTS)
    ]
    return sorted(files, key=natural_key)


def load_lq(path, device):
    """Same preprocessing as scripts/inference.py: crop size down to a multiple of 8 via resize."""
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    new_w, new_h = w - w % 8, h - h % 8
    if (new_w, new_h) != (w, h):
        img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))

    t01 = torch.tensor(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    tm11 = t01 * 2.0 - 1.0
    return tm11, t01, img


def postprocess(out_t, lq_img, align_method):
    """out_t: (1, 3, H, W) in [-1, 1] -> PIL image (with optional color fix)."""
    out_01 = (out_t[0].detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
    out_pil = transforms.ToPILImage()(out_01)

    if align_method == "adain":
        out_pil = adain_color_fix(target=out_pil, source=lq_img)
    elif align_method == "wavelet":
        out_pil = wavelet_color_fix(target=out_pil, source=lq_img)
    return out_pil


# =========================
# Main
# =========================
def main():
    args = parse_args()
    device = "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frames = list_frames(args.input_dir)
    if not frames:
        print(f"[ERROR] No images found in {args.input_dir}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Mode] use_intra={args.use_intra}, use_inter={args.use_inter}, "
          f"num_views={args.num_views}, use_spatial_r={args.use_spatial_r}")
    print(f"[Input] {len(frames)} frames from {args.input_dir}")

    model = CompMVR_test(args).to(device).eval()

    comp = CPE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="BR")
    comp.load_state_dict(torch.load(args.comp_path, map_location=device))
    comp.to(device).eval()

    V = args.num_views
    for i in tqdm(range(0, len(frames), V), desc="restore"):
        chunk = frames[i:i + V]
        real_count = len(chunk)
        while len(chunk) < V:          # pad last chunk by repeating the last frame
            chunk.append(chunk[-1])

        lqs, lq01s, lq_imgs = [], [], []
        for p in chunk:
            lq, lq01, lq_img = load_lq(p, device)
            lqs.append(lq)
            lq01s.append(lq01)
            lq_imgs.append(lq_img)

        shapes = {t.shape[-2:] for t in lqs}
        if len(shapes) > 1:
            names = [op.basename(p) for p in chunk[:real_count]]
            raise ValueError(f"Frames in one chunk must share a resolution: {names} -> {shapes}")

        lq_flat = torch.cat(lqs, dim=0)      # [V, 3, H, W], [-1, 1]
        lq01_flat = torch.cat(lq01s, dim=0)  # [V, 3, H, W], [0, 1]

        with torch.no_grad():
            emb = comp.get_visual_embedding(lq01_flat)
            model._current_real_views = real_count
            out_flat = model(lq_flat, emb)   # [V, 3, H, W], [-1, 1]

        for v in range(real_count):
            out_pil = postprocess(out_flat[v].unsqueeze(0), lq_imgs[v], args.align_method)
            out_pil.save(op.join(args.output_dir, op.basename(chunk[v])))

    print(f"=== DONE === {len(frames)} frames -> {args.output_dir}")


if __name__ == "__main__":
    main()

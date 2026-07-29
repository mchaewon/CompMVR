"""
main_test_unified.py
"""

import os
import re
import csv
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import pyiqa

from diffusion.CompMVR import CompMVR_test
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from cpe.cpe import CPE
from peft import LoraConfig
from data.dataset import MultiViewDataset, multiview_collate_fn, Dataset
from torch.utils.data import DataLoader
from utils import utils_option as option
# ══════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model",  type=str, required=True)
    parser.add_argument("--compmvr_path",    type=str, required=True)
    parser.add_argument("--comp_path",         type=str, required=True)
    parser.add_argument("--datasets",      type=str, default="./options/config.json")
    parser.add_argument("--save_root", type=str,
                        default="./results")

    parser.add_argument("--lora_rank",      type=int,   default=16)
    parser.add_argument("--timestep",       type=int,   default=499)
    parser.add_argument("--mixed_precision",type=str,   default="fp16",
                        choices=["fp16", "fp32"])

    parser.add_argument("--align_method",   type=str,   default="adain",
                        choices=["wavelet", "adain", "nofix"])
    parser.add_argument("--seed",           type=int,   default=123)
    parser.add_argument("--merge_and_unload_lora", action="store_true", default=False)

    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size",      type=int, default=96)
    parser.add_argument("--latent_tiled_overlap",   type=int, default=32)

    parser.add_argument("--use_intra",     action="store_true", default=False)
    parser.add_argument("--use_inter",     action="store_true", default=False)
    parser.add_argument("--num_views",      type=int,   default=1)
    
    parser.add_argument("--use_spatial_r", action="store_true", default=False)


    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Utils
# ══════════════════════════════════════════════════════════════════════

def parse_dir_info(comp_dir):
    qp_match = re.search(r'_QP(\d+)$', comp_dir)
    qp = int(qp_match.group(1)) if qp_match else None

    base = re.sub(r'_QP\d+$', '', comp_dir)  
    parts = base.split('_')                     

    codec = parts[0] if len(parts) >= 1 else "unknown"

    num_views = parts[-1] if parts[-1].isdigit() else "1"

    scene_parts = []
    for p in parts[1:]:
        if p.isdigit():
            break
        scene_parts.append(p)
    scene = '_'.join(scene_parts) if scene_parts else "unknown"

    gt_scene_dir = f"{scene}_{num_views}"

    return codec, scene, num_views, qp, gt_scene_dir

def get_images(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")))


def extract_frame_id(path):
    match = re.search(r'frame_(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else None


def make_vis(gt, comp, out):
    """GT / LQ / Restored / Error 4-panel visualization."""
    err = np.abs(out - gt)
    err = err / (err.max() + 1e-8)

    def to_u8(x):
        return (x * 255).clip(0, 255).astype(np.uint8)

    return np.concatenate([to_u8(gt), to_u8(comp), to_u8(out), to_u8(err)], axis=1)


def init_csv(path, header):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


# ══════════════════════════════════════════════════════════════════════
# Model Load
# ══════════════════════════════════════════════════════════════════════

def load_model(args):
    model = CompMVR_test(args)
    return model


def _flush_scene(scene, scene_scores, scene_csv, metric_keys, all_results):
    scene_avg = [float(np.mean(scene_scores[k])) for k in metric_keys]
    with open(scene_csv, "a", newline="") as f:
        csv.writer(f).writerow([scene] + scene_avg)
    print(f"  [{scene}] " + " | ".join(f"{k}={v:.4f}" for k, v in zip(metric_keys, scene_avg)))
    all_results.append(scene_avg)
    

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    frame_csv   = os.path.join(args.save_root, "frame.csv")
    scene_csv   = os.path.join(args.save_root, "scene.csv")
    overall_csv = os.path.join(args.save_root, "overall.csv")

    metric_keys = ["psnr", "msssim", "lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa"]

    init_csv(frame_csv,   ["scene","frame"] + metric_keys)
    init_csv(scene_csv,   ["scene"] + metric_keys)

    model = load_model(args)

    comp = CPE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
    comp.load_state_dict(torch.load(args.comp_path))
    comp.eval().cuda()

    device = "cuda"
    metrics = {
        "psnr":   pyiqa.create_metric("psnr",      device=device),
        "msssim": pyiqa.create_metric("ms_ssim",   device=device),
        "lpips":  pyiqa.create_metric("lpips-vgg", device=device),
        "dists":  pyiqa.create_metric("dists",     device=device),
        "niqe":   pyiqa.create_metric("niqe",      device=device),
        "musiq":  pyiqa.create_metric("musiq",     device=device),
        "maniqa": pyiqa.create_metric("maniqa",    device=device),
        "clipiqa":pyiqa.create_metric("clipiqa+",  device=device),
    }
    fr_keys = {"psnr", "msssim", "lpips", "dists"}
    nr_keys = {"niqe", "musiq", "maniqa", "clipiqa"}

    tensor_transforms = transforms.Compose([transforms.ToTensor()])

    args.datasets = option.parse_dataset(args.datasets)['datasets']
    test_dataset_opt = args.datasets.get('test')

    if args.use_inter:
        test_set = Dataset(test_dataset_opt)
        test_set.normalize = True
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False,
            num_workers=0, drop_last=False, pin_memory=True)
    else:
        test_set = Dataset(test_dataset_opt)
        test_set.normalize = True
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False,
            num_workers=1, drop_last=False, pin_memory=True)

    all_results = []
    current_scene = None
    scene_buffer = {}  
    for batch in tqdm(test_loader):
        lq     = batch['L'].cuda()
        gt     = batch['H'].cuda()
        lq_arr = batch['img_L_array'].cuda()
        scene  = batch['scene'][0] if isinstance(batch['scene'], list) else batch['scene']
        fid    = batch['frame_id']

        if scene not in scene_buffer:
            scene_buffer[scene] = []
        scene_buffer[scene].append((lq, lq_arr, gt, fid))

    for scene, frames in scene_buffer.items():
        scene_scores = {k: [] for k in metric_keys}

        os.makedirs(os.path.join(args.save_root, scene, "frame"), exist_ok=True)
        
        for i in range(0, len(frames), args.num_views):
            chunk = frames[i:i + args.num_views]
            
            while len(chunk) < args.num_views:
                chunk.append(chunk[-1])
            
            lq_flat     = torch.cat([f[0] for f in chunk], dim=0)  # (N, C, H, W)
            lq_arr_flat = torch.cat([f[1] for f in chunk], dim=0)
            gt_flat     = torch.cat([f[2] for f in chunk], dim=0)
            fids        = [f[3] for f in chunk]
            if lq_arr_flat.dtype == torch.uint8:
                lq_arr_flat = lq_arr_flat.float() / 255.0
            if lq_arr_flat.shape[-1] == 3:  
                lq_arr_flat = lq_arr_flat.permute(0, 3, 1, 2).contiguous() 
            
            N, C, H, W = lq_flat.shape
            B = 1  
            real_count = min(len(frames) - i, args.num_views)
            with torch.no_grad():
                visual_emb = comp.get_visual_embedding(lq_arr_flat)
                out_flat   = model(lq_flat, visual_emb)  # (N, C, H, W)
            out_views = out_flat.unsqueeze(0)   # (1, N, C, H, W)
            gt_views  = gt_flat.unsqueeze(0)    # (1, N, C, H, W)

            
            for v_idx in range(real_count):
                out_t = out_views[0, v_idx].unsqueeze(0)   # (1, C, H, W)
                gt_t  = gt_views[0, v_idx].unsqueeze(0)    # (1, C, H, W)

                out_01 = out_t * 0.5 + 0.5
                gt_01  = gt_t  * 0.5 + 0.5
                lq_01  = (lq_flat[v_idx].unsqueeze(0) * 0.5 + 0.5
                        if args.use_inter else lq * 0.5 + 0.5)

                # color fix
                out_pil = transforms.ToPILImage()(out_01[0].cpu())
                lq_np   = (lq_01[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

                if args.align_method == "adain":
                    out_pil = adain_color_fix(target=out_pil, source=lq_np)
                elif args.align_method == "wavelet":
                    out_pil = wavelet_color_fix(target=out_pil, source=lq_np)

                out_t_fixed = transforms.ToTensor()(out_pil).unsqueeze(0).cuda()

                frame_result = {}
                for k, fn in metrics.items():
                    val = fn(out_t_fixed, gt_01).item() if k in fr_keys else fn(out_t_fixed).item()
                    frame_result[k] = val
                    scene_scores[k].append(val)

                fid = fids[v_idx]
                if isinstance(fid, torch.Tensor):
                    fid = fid.item()
                img_name = f"frame_{int(fid):04d}.png"
                save_path = os.path.join(args.save_root, scene, "frame", img_name)
                Image.fromarray(np.array(out_pil)).save(os.path.join(save_path))


                with open(frame_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [scene, img_name] + [frame_result[k] for k in metric_keys])

        _flush_scene(scene, scene_scores, scene_csv, metric_keys, all_results)

    # overall
    if all_results:
        overall = np.mean(np.array(all_results), axis=0).tolist()
        with open(overall_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metric_keys)
            writer.writerow([f"{v:.4f}" for v in overall])
        print("\n=== Overall ===")
        for k, v in zip(metric_keys, overall):
            print(f"  {k:8s}: {v:.4f}")
    


if __name__ == "__main__":
    main()

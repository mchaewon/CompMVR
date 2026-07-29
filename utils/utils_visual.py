

import os
import numpy as np
import torch
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms



def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """[1,4,H,W] or [4,H,W] → float32 numpy [4,H,W]"""
    t = t.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    return t.numpy()


def _norm_uint8(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def latent_to_pca_rgb(latent: torch.Tensor, upscale: int = 4) -> Image.Image:

    feat = _to_numpy(latent)             # [4, H, W]
    C, H, W = feat.shape

    flat     = feat.reshape(C, H * W).T  # [H*W, 4]
    centered = flat - flat.mean(axis=0)

    _, vecs = np.linalg.eigh(np.cov(centered.T))  
    components = vecs[:, -3:]                      

    proj = (centered @ components).reshape(H, W, 3)
    rgb  = np.stack([_norm_uint8(proj[:, :, i]) for i in range(3)], axis=-1)

    img = Image.fromarray(rgb, mode="RGB")
    if upscale > 1:
        img = img.resize((W * upscale, H * upscale), Image.NEAREST)
    return img


def save_latent_viz(
    z_L:        torch.Tensor,
    model_pred: torch.Tensor,
    x_denoised: torch.Tensor,
    out_dir:    str = "results/latent_viz",
    step:       int = 0,
    upscale:    int = 4,
):

    os.makedirs(out_dir, exist_ok=True)

    imgs = {
        "z_L":        latent_to_pca_rgb(z_L,        upscale),
        "model_pred": latent_to_pca_rgb(model_pred,  upscale),
        "x_denoised": latent_to_pca_rgb(x_denoised,  upscale),
    }

    for name, img in imgs.items():
        img.save(os.path.join(out_dir, f"step{step:06d}_{name}.png"))

    W_img, H_img = imgs["z_L"].size
    grid = Image.new("RGB", (W_img * 3, H_img))
    for i, img in enumerate(imgs.values()):
        grid.paste(img, (W_img * i, 0))
    grid_path = os.path.join(out_dir, f"step{step:06d}_grid.png")
    grid.save(grid_path)

def visualize_r_gt(c_t: torch.Tensor,
                   c_tgt: torch.Tensor,
                   save_path: str = None,
                   idx: int = 0):

    with torch.no_grad():
        r_gt = (c_tgt - c_t)[idx]  # (3, H, W) [-2, 2]
    
    # (3, H, W) → (H, W, 3) numpy
    r_np = r_gt.cpu().float().permute(1, 2, 0).numpy()  # (H, W, 3)

    r_mean = r_np.mean(axis=2)  # (H, W)

    r_pos = np.clip(r_mean, 0, None)  
    r_neg = np.clip(-r_mean, 0, None) 
    
    lq_img  = (c_t[idx].cpu().float() * 0.5 + 0.5).permute(1,2,0).numpy().clip(0,1)
    hq_img  = (c_tgt[idx].cpu().float() * 0.5 + 0.5).permute(1,2,0).numpy().clip(0,1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(lq_img)
    axes[0, 0].set_title('LQ ', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(hq_img)
    axes[0, 1].set_title('HQ ', fontsize=12)
    axes[0, 1].axis('off')
    
    vmax = max(abs(r_mean.max()), abs(r_mean.min()), 0.01)
    im = axes[0, 2].imshow(r_mean, cmap='RdBu_r', 
                            vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('r_gt = HQ - LQ', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im_pos = axes[1, 0].imshow(r_pos, cmap='Reds', vmin=0, vmax=vmax)
    axes[1, 0].axis('off')
    plt.colorbar(im_pos, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im_neg = axes[1, 1].imshow(r_neg, cmap='Blues', vmin=0, vmax=vmax)
    axes[1, 1].axis('off')
    plt.colorbar(im_neg, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    pos_sum = r_pos.sum()
    neg_sum = r_neg.sum()
    total   = pos_sum + neg_sum + 1e-8
    
    axes[1, 2].pie(
        [pos_sum / total, neg_sum / total],
        labels=[f'\n{pos_sum/total*100:.1f}%',
                f'artifact \n{neg_sum/total*100:.1f}%'],
        colors=['#ff6b6b', '#6b9fff'],
        autopct='%1.1f%%',
        startangle=90,
    )
    axes[1, 2].set_title('ratio', fontsize=12)
    
    fig.suptitle(
        f'r_gt | mean={r_mean.mean():.4f} | '
        f'std={r_mean.std():.4f} | '
        f'pos_ratio={pos_sum/total*100:.1f}%',
        fontsize=13, y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_r_gt_batch(c_t: torch.Tensor,
                          c_tgt: torch.Tensor,
                          save_dir: str = "./r_gt_vis",
                          max_samples: int = 4):

    os.makedirs(save_dir, exist_ok=True)
    B = min(c_t.shape[0], max_samples)
    
    for i in range(B):
        visualize_r_gt(
            c_t, c_tgt,
            save_path=os.path.join(save_dir, f"r_gt_sample_{i}.png"),
            idx=i
        )


def visualize_residual(residual: torch.Tensor,
                       save_path: str,
                       idx: int = 0):

    with torch.no_grad():
        if residual.dim() == 4:
            r = residual[idx].cpu().float()
        else:
            r = residual.cpu().float()

        # [-1,1] or [-2,2] → [0,1] normalize
        r_min = r.min()
        r_max = r.max()
        r_vis = (r - r_min) / (r_max - r_min + 1e-8)  # (3, H, W) [0,1]

        r_np = (r_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(r_np).save(save_path)
    print(f"[Saved] {save_path}")
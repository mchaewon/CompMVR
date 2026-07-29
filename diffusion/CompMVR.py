import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'diffusion'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusion.models.autoencoder_kl import AutoencoderKL
from diffusion.models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from typing import Optional


import numpy as np
from numpy import pi, exp, sqrt

from diffusion.my_utils.vaehook import VAEHook


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
              "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and "encoder" in n:
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif "quant_conv" in n and "post_quant_conv" not in n:
                l_target_modules_encoder.append(n.replace(".weight", ""))

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    return vae, l_target_modules_encoder


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    
    if getattr(args, 'use_spatial_r', False) and getattr(args, 'use_intra', True):
        old_conv = unet.conv_in  
        new_conv = nn.Conv2d(8, 320, kernel_size=3, padding=1)
        nn.init.zeros_(new_conv.weight)
        new_conv.weight.data[:, :4, :, :] = old_conv.weight.data.clone()
        new_conv.bias.data = old_conv.bias.data.clone()
        unet.conv_in = new_conv
        print("[SpatialR] conv_in 4ch → 8ch ")

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in",
              "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj",
              "downsamplers.0.conv", "upsamplers.0.conv"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                # UNet add 
                if "conv_in" in n and getattr(args, 'use_spatial_r', False):
                    break  
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    lora_conf_others  = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others,  adapter_name="default_others")
    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class CompressionArtifactEstimator(nn.Module):

    def __init__(self, in_ch=3, hidden_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch*2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch*2, hidden_ch*2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden_ch*2, hidden_ch*4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch*4, hidden_ch*4, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch*4, hidden_ch*2, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch*2, hidden_ch*2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch*2, hidden_ch, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(hidden_ch, in_ch, 3, padding=1),
            nn.Sigmoid(),  # [0,1]
        )

    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        """lq: (B*V, 3, H, W) [-1,1] → r_pred: (B*V, 3, H, W) [0,1]"""
        lq_01 = lq * 0.5 + 0.5
        e1 = self.enc1(lq_01)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3) + e2
        d1 = self.dec1(d2) + e1

        out = self.out(d1)
        return out


def compression_artifact_loss(r_pred: torch.Tensor,
                              r_gt: torch.Tensor,
                              alpha: float = 0.3) -> torch.Tensor:

    r_gt_norm  = r_gt / 2.0
    error      = r_pred - r_gt_norm
    is_under   = (error < 0).float()
    weight     = torch.abs(alpha - is_under)
    asymm_loss = (weight * error.pow(2)).mean()

    r_pred_n  = F.normalize(r_pred.flatten(1), dim=1)
    r_gt_n    = F.normalize(r_gt_norm.flatten(1), dim=1)
    corr_loss = (1 - (r_pred_n * r_gt_n).sum(dim=1)).mean()
    return asymm_loss + 0.1 * corr_loss

class TemporalConvLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0)),
        )
        nn.init.zeros_(self.conv2[-1].weight)
        nn.init.zeros_(self.conv2[-1].bias)

    def forward(self, hidden_states: torch.Tensor, num_views: int) -> torch.Tensor:

        residual = hidden_states # (8, 320, 32, 32)

        B_V, C, H, W = hidden_states.shape
        B = B_V // num_views
        
        # (B*V,C,H,W) → (B, V, C, H, W) -> (B,C,V,H,W)= (2, 320, 4, 32, 32)
        hs = hidden_states.reshape(B, num_views, C, H, W).permute(0, 2, 1, 3, 4) 
        #conv3d 
        hs = self.conv2(self.conv1(hs)) 
        # (B,C,V,H,W) -> (B, V, C, H, W) -> (B*V, C, H, W)
        hs = hs.permute(0, 2, 1, 3, 4).reshape(B_V, C, H, W)  # (B*V, C, H, W) = (8, 320, 32, 32)

        return hs + residual

class Spatial3DResBlock(nn.Module):
    def __init__(self, channels: int, num_views: int, dropout: float = 0.0):
        super().__init__()
        self.num_views     = num_views
        self.temporal_conv = TemporalConvLayer(channels, channels, dropout)
        self.mix_factor    = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_out: torch.Tensor) -> torch.Tensor:
        alpha        = torch.sigmoid(self.mix_factor)
        temporal_out = self.temporal_conv(spatial_out, self.num_views)
        out = alpha * spatial_out + (1 - alpha) * temporal_out
        return out # (B*V, C, H, W) = (8, 320, 32, 32)

    def load_svd_weights(self, svd_state_dict: dict, block_key: str):
        prefix  = f"{block_key}.temporal_conv."
        matched = {k[len(prefix):]: v for k, v in svd_state_dict.items()
                   if k.startswith(prefix)}
        if matched:
            m, u = self.temporal_conv.load_state_dict(matched, strict=False)
            print(f"[SVD] {block_key}: {len(matched)} keys, missing={len(m)}, unexpected={len(u)}")
        else:
            print(f"[SVD] {block_key}: no keys found, zero-init")


class ViewConsistency3DAttention(nn.Module):
    def __init__(self, query_dim: int, num_heads: int, head_dim: int,
                 num_views: int, dropout: float = 0.0):
        super().__init__()
        self.num_views = num_views
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale     = head_dim ** -0.5

        self.norm   = nn.LayerNorm(query_dim)
        self.to_q   = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k   = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_v   = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        self.r_bias_proj = nn.Linear(3, 1)

        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)
        nn.init.zeros_(self.r_bias_proj.weight)
        nn.init.zeros_(self.r_bias_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B*V, HW, C)
        r_pred:        (B*V, 3, H, W) [0,1] or None
        Returns:       (B*V, HW, C)
        """
        # (B*V, HW, C) = (8, 1024, 320)
        B_V, HW, C = hidden_states.shape
        B = B_V // self.num_views
        V = self.num_views

        x = self.norm(hidden_states).reshape(B, V * HW, C) # (B, V*HW, C) = (2, 4096, 320)
        

        Q      = self.to_q(x) # (B, V*HW, inner_dim) = (2, 4096, 320)
        K      = self.to_k(x) # (B, V*HW, inner_dim)
        V_proj = self.to_v(x) # (B, V*HW, inner_dim)

        def split_heads(t):
            return t.reshape(B, V*HW, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V_proj = split_heads(Q), split_heads(K), split_heads(V_proj)
        attn_scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V_proj)
        out = out.transpose(1, 2).reshape(B, V * HW, self.inner_dim)
        out = self.to_out(out).reshape(B_V, HW, C) # (B*V, HW, C) = (8, 1024, 320)

        return hidden_states + out  

class VGGTCorrespondenceLoss(nn.Module):
 
    def __init__(
        self,
        # conf_threshold: float = 0.1,
        max_pairs: int = 4,
        subsample: int = 4
    ):
        super().__init__()
        # self.conf_threshold = conf_threshold
        self.max_pairs = max_pairs
        self.subsample = subsample
 
    def forward(
        self,
        restored_imgs: list[torch.Tensor],  # list of (B, 3, H, W)
        hq_imgs: list[torch.Tensor],        # list of (B, 3, H, W)
        vggt_model: nn.Module,              # frozen VGGT model
    ) -> torch.Tensor:
        N = len(restored_imgs)
        if N < 2:
            return torch.tensor(0.0, device=restored_imgs[0].device)
 
        B = restored_imgs[0].shape[0]
        device = restored_imgs[0].device
 
        with torch.no_grad():
            hq_stack = torch.stack(hq_imgs, dim=1)  # (B, N, 3, H, W)
            
            B, N, C, H, W = hq_stack.shape
            hq_stack_vggt = F.interpolate(
                hq_stack.view(B*N, C, H, W),
                size=(518, 518),
                mode='bilinear', align_corners=False
            ).view(B, N, C, 518, 518)

            predictions = vggt_model(hq_stack_vggt) 
 
            world_points = predictions["world_points"]       # (B, N, H_v, W_v, 3)
            world_conf   = predictions["world_points_conf"]  # (B, N, H_v, W_v)
            
        H_v, W_v = world_points.shape[2], world_points.shape[3]
        restored_resized = []
        for img in restored_imgs:
            restored_resized.append(
                F.interpolate(img, size=(H_v, W_v), mode="bilinear", align_corners=False)
            )  # (B, 3, H_v, W_v)

        hq_resized = []
        for img in hq_imgs:
            hq_resized.append(
                F.interpolate(img, size=(H_v, W_v), mode="bilinear", align_corners=False)
            )
 
        loss_total = torch.tensor(0.0, device=device)
        pair_count = 0
 
        s = self.subsample
 
        for i in range(N):
            for j in range(i + 1, N):
                if pair_count >= self.max_pairs:
                    break
 
                l_ij = self._pair_loss(
                    restored_resized[i],
                    restored_resized[j],
                    hq_resized[i],
                    hq_resized[j],
                    world_points[:, i],
                    world_points[:, j],
                    world_conf[:, i],
                    world_conf[:, j],
                    s,
                )
                l_ji = self._pair_loss(
                    restored_resized[j],
                    restored_resized[i],
                    hq_resized[j],
                    hq_resized[i],
                    world_points[:, j],
                    world_points[:, i],
                    world_conf[:, j],
                    world_conf[:, i],
                    s,
                )

                l = 0.5 * (l_ij + l_ji)

                loss_total = loss_total + l
                pair_count += 1
 
        if pair_count == 0:
            return torch.tensor(0.0, device=device)
 
        return loss_total / pair_count
 
    def _pair_loss(
        self,
        img_i: torch.Tensor,    # (B, 3, H, W)
        img_j: torch.Tensor,    # (B, 3, H, W)
        gt_i: torch.Tensor,
        gt_j: torch.Tensor,
        pts_i: torch.Tensor,    # (B, H, W, 3) world coords
        pts_j: torch.Tensor,    # (B, H, W, 3) world coords
        conf_i: torch.Tensor,   # (B, H, W)
        conf_j: torch.Tensor,   # (B, H, W)
        subsample: int,
    ) -> torch.Tensor:
        B, _, H, W = img_i.shape
        device = img_i.device
 
        # (B, 3, H, W) → (B, 3, H//s, W//s)
        img_i_s  = img_i[:, :, ::subsample, ::subsample]
        img_j_s  = img_j[:, :, ::subsample, ::subsample]
        gi_s = gt_i[:, :, ::subsample, ::subsample]
        gj_s = gt_j[:, :, ::subsample, ::subsample]
        pts_i_s  = pts_i[:, ::subsample, ::subsample, :]  # (B, Hs, Ws, 3)
        pts_j_s  = pts_j[:, ::subsample, ::subsample, :]
        conf_i_s = conf_i[:, ::subsample, ::subsample]    # (B, Hs, Ws)
        conf_j_s = conf_j[:, ::subsample, ::subsample]
 
        Hs, Ws = pts_i_s.shape[1], pts_i_s.shape[2]

        pts_i_flat  = pts_i_s.reshape(B, Hs * Ws, 3)
        pts_j_flat  = pts_j_s.reshape(B, Hs * Ws, 3)
        conf_i_flat = conf_i_s.reshape(B, Hs * Ws)
        conf_j_flat = conf_j_s.reshape(B, Hs * Ws)
        img_i_flat  = img_i_s.reshape(B, 3, Hs * Ws).permute(0, 2, 1)
        img_j_flat  = img_j_s.reshape(B, 3, Hs * Ws).permute(0, 2, 1)
        gi_flat = gi_s.reshape(B, 3, Hs * Ws).permute(0, 2, 1).detach()
        gj_flat = gj_s.reshape(B, 3, Hs * Ws).permute(0, 2, 1).detach()

 
        threshold_i = 0.75 * conf_i_flat.mean(dim=1, keepdim=True)
        threshold_j = 0.75 * conf_j_flat.mean(dim=1, keepdim=True)
        valid_i = conf_i_flat > threshold_i   # (B, P)
        valid_j = conf_j_flat > threshold_j   # (B, Q)
 
        
        with torch.no_grad():
            dist_raw = torch.cdist(pts_i_flat.float(), pts_j_flat.float(), p=2)  # (B, P, Q)

            dist_masked = dist_raw + (~valid_j).unsqueeze(1).float() * 1e6

            match_idx_ij = dist_masked.argmin(dim=2)   # (B, P)
            match_idx_ji = dist_masked.argmin(dim=1)   # (B, Q)

            min_dist = dist_raw.gather(2, match_idx_ij.unsqueeze(2)).squeeze(2)  # (B, P)
            threshold = min_dist.mean(dim=1, keepdim=True) * 1.5
            geo_mask = min_dist < threshold
 

        B, P = match_idx_ij.shape
        idx_p = torch.arange(P, device=device).unsqueeze(0).expand(B, P)

        mutual_mask = (
            match_idx_ji.gather(1, match_idx_ij) == idx_p
        )  # (B, P)

        
        conf_j_matched = conf_j_flat.gather(1, match_idx_ij)
        match_conf = conf_i_flat * conf_j_matched

        valid_j_matched = valid_j.gather(1, match_idx_ij)

        
        final_mask = (
            valid_i &
            valid_j_matched &
            mutual_mask &
            geo_mask
        )
        
        
        match_conf = match_conf * final_mask.float()

        k_ratio = 0.1
        k = max(1, int(match_conf.shape[1] * k_ratio))

        topk_val, topk_idx = torch.topk(match_conf, k, dim=1)

        topk_mask = torch.zeros_like(match_conf)
        topk_mask.scatter_(1, topk_idx, 1.0)


        final_mask = final_mask & (topk_mask > 0)
        
       
        match_idx_expand = match_idx_ij.unsqueeze(2).expand(-1, -1, 3)

        gt_j_matched = gj_flat.gather(1, match_idx_expand)
        restored_j_matched = img_j_flat.gather(1, match_idx_expand)

        l1_cross_gt = (img_i_flat - gt_j_matched).abs().mean(dim=2)

        l1_restored_cons = (img_i_flat - restored_j_matched).abs().mean(dim=2)

        l1 = l1_cross_gt + 0.2 * l1_restored_cons

        weight = match_conf * final_mask.float()
        denom = weight.sum(dim=1).clamp(min=1e-8)
        loss_per_batch = (weight * l1).sum(dim=1) / denom

        return loss_per_batch.mean()
 
def _get_resnet_channels(module) -> int:
    for attr in ('out_channels', 'conv2'):
        m = getattr(module, attr, None)
        if m is None:
            continue
        if hasattr(m, 'out_channels'):
            return m.out_channels
        if isinstance(m, nn.Conv2d):
            return m.out_channels
        if isinstance(m, (nn.Sequential, nn.ModuleList)):
            for layer in (m if isinstance(m, nn.Sequential) else m):
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
    for attr in ('norm2', 'norm1'):
        m = getattr(module, attr, None)
        if m is not None and hasattr(m, 'num_channels'):
            return m.num_channels
        if m is not None and hasattr(m, 'num_features'):
            return m.num_features
    return 320


class CompMVR(nn.Module):

    def __init__(self, args, num_views: int = 1):
        super().__init__()
        self.args       = args
        self.num_views  = args.num_views
        self.use_intra  = args.use_intra
        self.use_inter  = args.use_inter

        print(f"check - self.use_intra : {self.use_intra}, self.use_inter : {self.use_inter}")

        # VAE / UNet
        self.vae, self.lora_vae_modules_encoder = initialize_vae(args)
        self.lora_rank_vae = args.lora_rank
        self.unet, self.lora_unet_modules_encoder, \
            self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
        self.lora_rank_unet = args.lora_rank
        self.unet.to("cuda")
        self.vae.to("cuda")

        # UNet add
        self.use_spatial_r = args.use_spatial_r
        self.r_spatial_adapter = None
        self.vae_r = None  

        if self.use_intra and self.use_spatial_r:
            self.r_spatial_adapter = nn.Conv2d(3, 4, kernel_size=1).to("cuda")
            nn.init.zeros_(self.r_spatial_adapter.weight)
            nn.init.zeros_(self.r_spatial_adapter.bias)
            print("[SpatialR] r_spatial_adapter (Conv 1x1)")
        else:
            self.use_vae_r = False


        timestep = getattr(args, 'timestep', 499)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.timesteps = torch.tensor([timestep], device="cuda").long()

        # Stage 1 visual embedding projection
        self.proj = nn.Linear(512, 1024).to("cuda")
        self.proj.requires_grad_(True)

        self.artifact_estimator = CompressionArtifactEstimator(in_ch=3, hidden_ch=32).to("cuda")

        if self.use_inter:
            self._build_inter_modules()
        else:
            self.temporal_res_blocks = nn.ModuleList()
            self.view_attn_blocks    = nn.ModuleList()

        self._hook_handles:   list                    = []
        self._hooks_active:   bool                    = False

        self._current_real_views = self.num_views 


    def _build_inter_modules(self):
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.attention import BasicTransformerBlock

        resnet_blocks, attn_blocks = [], []
        for name, module in self.unet.named_modules():
            if isinstance(module, ResnetBlock2D):
                resnet_blocks.append((name, _get_resnet_channels(module)))
            elif isinstance(module, BasicTransformerBlock):
                qd = (module.norm1.normalized_shape[0]
                      if hasattr(module.norm1, 'normalized_shape') else 320)
                attn_blocks.append((name, qd))

        self.temporal_res_blocks = nn.ModuleList([
            Spatial3DResBlock(channels=ch, num_views=self.num_views).to("cuda")
            for _, ch in resnet_blocks
        ])
        self._resnet_block_names = [n for n, _ in resnet_blocks]

        self.view_attn_blocks = nn.ModuleList([
            ViewConsistency3DAttention(
                query_dim=qd,
                num_heads=max(1, qd // 64),
                head_dim=min(qd, 64),
                num_views=self.num_views,
            ).to("cuda")
            for _, qd in attn_blocks
        ])
        self._attn_block_names = [n for n, _ in attn_blocks]

        print(f"[ViewInterModel] ResNet hooks: {len(resnet_blocks)}, "
              f"Attn hooks: {len(attn_blocks)}")


    def _register_hooks(self):
        if not self.use_inter:
            return
        self._remove_hooks()
        res_idx  = {n: i for i, n in enumerate(self._resnet_block_names)}
        attn_idx = {n: i for i, n in enumerate(self._attn_block_names)}

        for name, module in self.unet.named_modules():
            if name in res_idx:
                tb = self.temporal_res_blocks[res_idx[name]]
                self._hook_handles.append(
                    module.register_forward_hook(self._make_resnet_hook(tb)))
            elif name in attn_idx:
                ab = self.view_attn_blocks[attn_idx[name]]
                self._hook_handles.append(
                    module.register_forward_hook(self._make_attn_hook(ab)))

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _make_resnet_hook(self, temporal_block: Spatial3DResBlock):
        def hook(module, inp, out):
            if not self._hooks_active:
                return out
            return temporal_block(out)
        return hook

    def _make_attn_hook(self, attn_block: ViewConsistency3DAttention):
        def hook(module, inp, out):
            if not self._hooks_active:
                return out
            hidden   = out[0] if isinstance(out, tuple) else out
            enhanced = attn_block(hidden)
            return (enhanced,) + out[1:] if isinstance(out, tuple) else enhanced
        return hook


    def set_train(self):
        self.vae.train()
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.train()
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        if self.use_spatial_r:
            for p in self.unet.conv_in.parameters():
                p.requires_grad = True
        else:
            self.unet.conv_in.requires_grad_(True)


        if self.use_intra:
            self.artifact_estimator.train()
            for p in self.artifact_estimator.parameters():
                p.requires_grad = True
            if self.use_spatial_r and self.r_spatial_adapter is not None:
                self.r_spatial_adapter.train()
                for p in self.r_spatial_adapter.parameters():
                    p.requires_grad = True

        if self.use_inter:
            self._hooks_active = True
            self._register_hooks()
            self.temporal_res_blocks.train()  
            self.view_attn_blocks.train()  
            for p in self.temporal_res_blocks.parameters():
                p.requires_grad = True
            for p in self.view_attn_blocks.parameters():
                p.requires_grad = True

    def set_eval(self):
        self.vae.eval()
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = False
        self.unet.eval()
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = False
        self.unet.conv_in.requires_grad_(False)
        if self.use_intra:
            self.artifact_estimator.eval()
            for p in self.artifact_estimator.parameters():
                p.requires_grad = False
            if self.use_spatial_r and self.r_spatial_adapter is not None:
                self.r_spatial_adapter.eval()
                for p in self.r_spatial_adapter.parameters():
                    p.requires_grad = False
            
        if self.use_inter:
            self._hooks_active = False
            self.temporal_res_blocks.eval()
            self.view_attn_blocks.eval()

    def get_intra_params(self) -> list:
        params = []

        # UNet add
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                params.append(p)
        if self.use_spatial_r:
            for p in self.unet.conv_in.parameters():
                params.append(p)
        else:
            for n, p in self.unet.named_parameters():
                if "conv_in" in n:
                    params.append(p)

        # vae
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                params.append(p)
        params += list(self.proj.parameters())

        if self.use_intra:
            params += list(self.artifact_estimator.parameters())
            if self.use_spatial_r and self.r_spatial_adapter is not None:
                params += list(self.r_spatial_adapter.parameters())
        return params

    def get_inter_params(self) -> list:
        if not self.use_inter:
            return []
        return (list(self.temporal_res_blocks.parameters()) +
                list(self.view_attn_blocks.parameters()))

    def get_all_trainable_params(self) -> list:
        return self.get_intra_params() + self.get_inter_params()


    def _build_visual_embeds(self, c_t: torch.Tensor,
                              visual_embedding: torch.Tensor) -> torch.Tensor:


        if visual_embedding is not None:
            visual_embeds = self.proj(visual_embedding)  # (B*V, seq, 1024)
        else:
            B_V = c_t.shape[0]
            visual_embeds = torch.zeros(B_V, 1, 1024, device=c_t.device, dtype=c_t.dtype)

        return visual_embeds


    def forward(self, c_t: torch.Tensor,
                visual_embedding: torch.Tensor,
                c_tgt: Optional[torch.Tensor] = None):
        r_pred = self.artifact_estimator(c_t) if self.use_intra else None

        # 3. VAE encode
        z_L = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        if self.use_spatial_r and r_pred is not None:
            r_down   = F.adaptive_avg_pool2d(
                r_pred.detach(), (z_L.shape[2], z_L.shape[3]))
            r_latent = self.r_spatial_adapter(r_down)
            z_L_unet = torch.cat([z_L, r_latent], dim=1)
        else:
            z_L_unet = z_L

        visual_embeds = self._build_visual_embeds(c_t, visual_embedding)

        model_pred = self.unet(z_L_unet, self.timesteps,
                       encoder_hidden_states=visual_embeds).sample

        # 6. Denoising
        x_denoised = self.noise_scheduler.step(
            model_pred, self.timesteps, z_L, return_dict=True).prev_sample

        # 7. VAE decode → x_decoded
        x_decoded = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        output_image = x_decoded

        r_gt = None
        if c_tgt is not None:
            with torch.no_grad():
                r_gt = (c_tgt - c_t).abs()  # [0,2]

        return {
            "output_image": output_image,      
            "model_pred":   model_pred,
            "r_pred":       r_pred,
            "r_gt":         r_gt,
        }

    @torch.no_grad()
    def eval(self, lq: torch.Tensor, visual_embedding: torch.Tensor) -> torch.Tensor:
        r_pred = self.artifact_estimator(lq) if self.use_intra else None

        z_L = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor

        if self.use_spatial_r and r_pred is not None:
            r_down   = F.adaptive_avg_pool2d(
                r_pred.detach(), (z_L.shape[2], z_L.shape[3]))
            r_latent = self.r_spatial_adapter(r_down)
            z_L_unet = torch.cat([z_L, r_latent], dim=1)
        else:
            z_L_unet = z_L

        visual_embeds = self._build_visual_embeds(lq, visual_embedding)
        model_pred    = self.unet(z_L_unet, self.timesteps,
                              encoder_hidden_states=visual_embeds).sample
        x_denoised = self.noise_scheduler.step(
            model_pred, self.timesteps, z_L, return_dict=True).prev_sample
        x_decoded = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        return x_decoded


    def save_model(self, outf: str):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["rank_vae"]       = self.lora_rank_vae
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}

        sd["unet_lora_encoder_modules"] = self.lora_unet_modules_encoder
        sd["unet_lora_decoder_modules"] = self.lora_unet_modules_decoder
        sd["unet_lora_others_modules"]  = self.lora_unet_others
        sd["rank_unet"]       = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items()
                                 if "lora" in k or "conv_in" in k}

        sd["proj"]             = self.proj.state_dict()

        if self.use_intra:
            sd["artifact_estimator"] = self.artifact_estimator.state_dict()
            sd["use_intra"]       = self.use_intra

            if self.use_spatial_r and self.r_spatial_adapter is not None:
                sd["r_spatial_adapter"] = self.r_spatial_adapter.state_dict()
                sd["use_spatial_r"] = self.use_spatial_r

        if self.use_inter:
            sd["use_inter"] = self.use_inter
            sd["view_inter"] = {
                "temporal_res_blocks": self.temporal_res_blocks.state_dict(),
                "view_attn_blocks":    self.view_attn_blocks.state_dict(),
            }
            sd["num_views"] = self.num_views

        torch.save(sd, outf)
        print(f"[ViewInterModel] Saved: {outf}")

class CompMVR_test(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")

        self.vae  = AutoencoderKL.from_pretrained(
            args.pretrained_model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model, subfolder="unet")
        
        #UNet add
        self.use_spatial_r = args.use_spatial_r
        self.r_spatial_adapter = None

        self.proj = nn.Linear(512, 1024)

        # Intra
        self.artifact_estimator = CompressionArtifactEstimator(in_ch=3, hidden_ch=32)

        self.use_intra    = args.use_intra

        # Inter
        self.num_views = args.num_views
        self.use_inter = args.use_inter

        self._hook_handles:   list                    = []
        self._hooks_active:   bool                    = False

        if self.use_inter:
            self._build_inter_modules()
        else:
            self.temporal_res_blocks = nn.ModuleList()
            self.view_attn_blocks    = nn.ModuleList()

        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        if args.compmvr_path is None:
            print("[WARN] Not exist compmvr_path")
        else:
            ckpt = torch.load(args.compmvr_path, map_location="cpu")
            self.load_ckpt(ckpt)

        if args.merge_and_unload_lora:
            print("===> MERGE LORA <===")
            self.vae  = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda",  dtype=self.weight_dtype)
        self.proj.to("cuda")
        self.artifact_estimator.to("cuda", dtype=self.weight_dtype)
        if self.r_spatial_adapter is not None:
            self.r_spatial_adapter.to("cuda", dtype=self.weight_dtype)
        
        if self.use_inter:
            self.temporal_res_blocks.to("cuda", dtype=self.weight_dtype)
            self.view_attn_blocks.to("cuda", dtype=self.weight_dtype)

        self.timesteps = torch.tensor([args.timestep], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = \
            self.noise_scheduler.alphas_cumprod.cuda()
        
        self._current_real_views = self.num_views 

        self._set_eval()


    def _build_inter_modules(self):
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.attention import BasicTransformerBlock

        resnet_blocks, attn_blocks = [], []
        for name, module in self.unet.named_modules():
            if isinstance(module, ResnetBlock2D):
                resnet_blocks.append((name, _get_resnet_channels(module)))
            elif isinstance(module, BasicTransformerBlock):
                qd = (module.norm1.normalized_shape[0]
                      if hasattr(module.norm1, 'normalized_shape') else 320)
                attn_blocks.append((name, qd))

        self.temporal_res_blocks = nn.ModuleList([
            Spatial3DResBlock(channels=ch, num_views=self.num_views)
            for _, ch in resnet_blocks
        ])
        self._resnet_block_names = [n for n, _ in resnet_blocks]

        self.view_attn_blocks = nn.ModuleList([
            ViewConsistency3DAttention(
                query_dim=qd,
                num_heads=max(1, qd // 64),
                head_dim=min(qd, 64),
                num_views=self.num_views,
            )
            for _, qd in attn_blocks
        ])
        self._attn_block_names = [n for n, _ in attn_blocks]


    def _register_hooks(self):
        if not self.use_inter:
            return
        self._remove_hooks()
        res_idx  = {n: i for i, n in enumerate(self._resnet_block_names)}
        attn_idx = {n: i for i, n in enumerate(self._attn_block_names)}

        for name, module in self.unet.named_modules():
            if name in res_idx:
                tb = self.temporal_res_blocks[res_idx[name]]
                self._hook_handles.append(
                    module.register_forward_hook(self._make_resnet_hook(tb)))
            elif name in attn_idx:
                ab = self.view_attn_blocks[attn_idx[name]]
                self._hook_handles.append(
                    module.register_forward_hook(self._make_attn_hook(ab)))

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _make_resnet_hook(self, temporal_block):
        def hook(module, inp, out):
            if not self._hooks_active:
                return out
            return temporal_block(out)
        return hook

    def _make_attn_hook(self, attn_block):
        def hook(module, inp, out):
            if not self._hooks_active:
                return out
            hidden   = out[0] if isinstance(out, tuple) else out
            enhanced = attn_block(hidden)
            return (enhanced,) + out[1:] if isinstance(out, tuple) else enhanced
        return hook

    def _build_visual_embeds(self, c_t, visual_embedding):
        if visual_embedding is not None:
            visual_embeds = self.proj(visual_embedding)  # (B*V, seq, 1024)
        else:
            B_V = c_t.shape[0]
            visual_embeds = torch.zeros(B_V, 1, 1024, device=c_t.device, dtype=c_t.dtype)

        return visual_embeds

    def load_ckpt(self, ckpt: dict):
        self.proj.load_state_dict(ckpt["proj"])

        if self.use_spatial_r:
            old_conv = self.unet.conv_in  
            new_conv = nn.Conv2d(8, 320, kernel_size=3, padding=1)
            nn.init.zeros_(new_conv.weight)
            new_conv.weight.data[:, :4, :, :] = old_conv.weight.data.clone()
            new_conv.bias.data = old_conv.bias.data.clone()
            self.unet.conv_in = new_conv

            self.r_spatial_adapter = nn.Conv2d(3, 4, kernel_size=1)
            nn.init.zeros_(self.r_spatial_adapter.weight)
            nn.init.zeros_(self.r_spatial_adapter.bias)
            print("[Load] conv_in 8ch + r_spatial_adapter")


        lora_conf_enc = LoraConfig(r=ckpt["rank_unet"], init_lora_weights="gaussian",
                                   target_modules=ckpt["unet_lora_encoder_modules"])
        lora_conf_dec = LoraConfig(r=ckpt["rank_unet"], init_lora_weights="gaussian",
                                   target_modules=ckpt["unet_lora_decoder_modules"])
        lora_conf_oth = LoraConfig(r=ckpt["rank_unet"], init_lora_weights="gaussian",
                                   target_modules=ckpt["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_enc, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_dec, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_oth, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if ("lora" in n or "conv_in" in n) and n in ckpt["state_dict_unet"]:
                p.data.copy_(ckpt["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        if "state_dict_vae" in ckpt and "vae_lora_encoder_modules" in ckpt:
            vae_lora_conf = LoraConfig(r=ckpt["rank_vae"], init_lora_weights="gaussian",
                                       target_modules=ckpt["vae_lora_encoder_modules"])
            self.vae.add_adapter(vae_lora_conf, adapter_name="default_encoder")
            for n, p in self.vae.named_parameters():
                if "lora" in n and n in ckpt["state_dict_vae"]:
                    p.data.copy_(ckpt["state_dict_vae"][n])
            self.vae.set_adapter(["default_encoder"])

        if "artifact_estimator" in ckpt:
            self.artifact_estimator.load_state_dict(ckpt["artifact_estimator"])
            print("[Load] artifact_estimator weights loaded")

        if "r_spatial_adapter" in ckpt and self.r_spatial_adapter is not None:
            self.r_spatial_adapter.load_state_dict(ckpt["r_spatial_adapter"])
            print("[Load] r_spatial_adapter loaded")

        if "view_inter" in ckpt and self.use_inter:
            vi = ckpt["view_inter"]
            self.temporal_res_blocks.load_state_dict(vi["temporal_res_blocks"])
            self.view_attn_blocks.load_state_dict(vi["view_attn_blocks"])
            print("[Load] Inter weights loaded")

        print(f"[Load] Checkpoint loaded")
        

    @torch.no_grad()
    def forward(self, lq, visual_embedding):
        """lq: (B*N, C, H, W) flat tensor"""
        r_pred = self.artifact_estimator(lq.to(self.weight_dtype)) if self.use_intra else None

        if self.use_inter:
            self._hooks_active   = True
            self._register_hooks()

        lq_latent = (self.vae.encode(lq.to(self.weight_dtype))
                     .latent_dist.sample() * self.vae.config.scaling_factor)
        
        if self.use_spatial_r and r_pred is not None and self.r_spatial_adapter is not None:
            r_down    = F.adaptive_avg_pool2d(
                r_pred, (lq_latent.shape[2], lq_latent.shape[3]))
            r_latent  = self.r_spatial_adapter(r_down.to(self.weight_dtype))
            lq_latent_unet = torch.cat([lq_latent, r_latent], dim=1)  # 8ch
        else:
            lq_latent_unet = lq_latent  # 4ch

        visual_embeds = self._build_visual_embeds(
            lq.to(self.weight_dtype), visual_embedding)
        visual_embeds = visual_embeds.to(self.unet.dtype)

        _, _, h, w   = lq_latent.size()
        tile_size    = self.args.latent_tiled_size
        tile_overlap = self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            model_pred = self.unet(
                lq_latent_unet, self.timesteps,
                encoder_hidden_states=visual_embeds).sample
        else:
            model_pred = self._tiled_unet_forward(
                lq_latent_unet, visual_embeds, tile_size, tile_overlap, h, w)

        if self.use_inter:
            self._hooks_active   = False
            self._remove_hooks()

        x_denoised = self.noise_scheduler.step(
            model_pred, self.timesteps, lq_latent,
            return_dict=True).prev_sample
        x_decoded = (self.vae.decode(
            x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample).clamp(-1, 1)

        return x_decoded


    def _init_tiled_vae(self,
                        encoder_tile_size: int = 256,
                        decoder_tile_size: int = 256,
                        fast_decoder: bool = False,
                        fast_encoder: bool = False,
                        color_fix: bool = False,
                        vae_to_gpu: bool = True):
        if not hasattr(self.vae.encoder, "original_forward"):
            setattr(self.vae.encoder, "original_forward", self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, "original_forward"):
            setattr(self.vae.decoder, "original_forward", self.vae.decoder.forward)
 
        self.vae.encoder.forward = VAEHook(
            self.vae.encoder, encoder_tile_size,
            is_decoder=False, fast_decoder=fast_decoder,
            fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            self.vae.decoder, decoder_tile_size,
            is_decoder=True, fast_decoder=fast_decoder,
            fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
 
    def _tiled_unet_forward(self, lq_latent_unet, visual_embeds,
                         tile_size, tile_overlap, h, w):
        tile_size    = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        grid_rows, cur_x = 0, 0
        while cur_x < lq_latent_unet.size(-1):
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1
        grid_cols, cur_y = 0, 0
        while cur_y < lq_latent_unet.size(-2):
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list, noise_preds = [], []
        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1: ofs_x = w - tile_size
                if col == grid_cols - 1: ofs_y = h - tile_size

                input_tile = lq_latent_unet[:, :,
                                    ofs_y:ofs_y + tile_size,
                                    ofs_x:ofs_x + tile_size]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols - 1:
                    model_out = self.unet(
                        torch.cat(input_list, dim=0),
                        self.timesteps,
                        encoder_hidden_states=visual_embeds).sample
                    input_list = []
                noise_preds.append(model_out)

        B = lq_latent_unet.shape[0]
        noise_pred   = torch.zeros((B, 4, h, w), device=lq_latent_unet.device)
        contributors = torch.zeros((B, 4, h, w), device=lq_latent_unet.device)

        tile_weights = self._gaussian_weights(tile_size, tile_size, 1, out_ch=4)

        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1: ofs_x = w - tile_size
                if col == grid_cols - 1: ofs_y = h - tile_size

                noise_pred[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] \
                    += noise_preds[row * grid_cols + col] * tile_weights
                contributors[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] \
                    += tile_weights
        return noise_pred / contributors
 
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        var      = 0.01
        midpoint = (tile_width - 1) / 2
        x_probs  = [exp(-(x - midpoint) ** 2 / (tile_width ** 2) / (2 * var))
                    / sqrt(2 * pi * var) for x in range(tile_width)]
        midpoint = tile_height / 2
        y_probs  = [exp(-(y - midpoint) ** 2 / (tile_height ** 2) / (2 * var))
                    / sqrt(2 * pi * var) for y in range(tile_height)]
        weights  = np.outer(y_probs, x_probs)

        if out_ch is None:
            out_ch = 8 if self.use_spatial_r else self.unet.config.in_channels
        return torch.tile(
            torch.tensor(weights, device=self.device),
            (nbatches, out_ch, 1, 1))

 
 
    def _set_eval(self):
        self.vae.eval()
        self.unet.eval()
        if self.use_intra:         
            self.artifact_estimator.eval()
        if self.use_inter:
            self.temporal_res_blocks.eval()
            self.view_attn_blocks.eval()
        for p in self.parameters():
            p.requires_grad_(False)
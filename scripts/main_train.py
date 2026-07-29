import os
import random
import re
import argparse
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from collections import OrderedDict
from peft import LoraConfig
import pyiqa
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
import warnings
warnings.filterwarnings("ignore")
import wandb
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import utils_image as util
from utils import utils_option as option
from utils.utils_visual import visualize_residual
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from diffusion.CompMVR import (
    CompMVR,
    compression_damage_loss,
    VGGTCorrespondenceLoss
)
from diffusion.models.discriminator import Discriminator
from vggt.models.vggt import VGGT

tensor_transforms = transforms.Compose([transforms.ToTensor()])


# ══════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_batch_size", type=int, default=2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="train_ours")
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--datasets", default='options/config.json')
    parser.add_argument('--comp_path', type=str, default=None)
    parser.add_argument('--gan_loss', action='store_true')
    parser.add_argument('--gan_dis_weight', default=1e-2, type=float)
    parser.add_argument('--gan_gen_weight', default=5e-3, type=float)
    parser.add_argument("--align_method", type=str,
                        choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--timestep", type=int, default=499)
    parser.add_argument("--val_num_samples", type=int, default=500)

    parser.add_argument("--use_intra", action="store_true", default=False,
                        help="Intra-view ")
    parser.add_argument("--use_inter", action="store_true", default=False,
                        help="Inter-view ")
    
    # Intra: r_pred
    parser.add_argument("--r_loss_weight", type=float, default=0.3)
    parser.add_argument("--r_loss_alpha",  type=float, default=0.3)

    # Inter: multi-view 3D conv/attention
    parser.add_argument("--num_views", type=int, default=1)

    # L_corr (Correspondence loss)
    parser.add_argument("--use_lcorr", action="store_true", default=False,
                        help="correspondence loss")
    parser.add_argument("--lambda_lcorr", type=float, default=0.1,
                        help="L_corr loss weight")
    parser.add_argument("--lcorr_conf_threshold", type=float, default=0.2,
                        help="confidence threshold")

    parser.add_argument("--use_spatial_r", action="store_true", default=False)

    # vis
    parser.add_argument("--resi_pred_vis", action="store_true")

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════
def validate_args(args):
    if not args.use_inter:
        args.num_views = 1

def get_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [(int(m.group(1)), d)
             for d in os.listdir(ckpt_dir)
             for m in [re.match(r"model_(\d+)", d)] if m]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return os.path.join(ckpt_dir, ckpts[-1][1]), ckpts[-1][0]


def custom_collate_fn(batch):
    for key in batch[0].keys():
        for i, item in enumerate(batch):
            if item[key] is None:
                print(f"[DEBUG] None found: key='{key}', batch_idx={i}")
    img_L_arrays = [item.pop('img_L_array') for item in batch]
    collated     = torch.utils.data.dataloader.default_collate(batch)
    img_L_arrays = [torch.tensor(x) for x in img_L_arrays]
    max_h = max(img.shape[0] for img in img_L_arrays)
    max_w = max(img.shape[1] for img in img_L_arrays)
    padded = []
    for img in img_L_arrays:
        h, w, c = img.shape
        pad = torch.zeros((max_h, max_w, c), dtype=img.dtype)
        pad[(max_h-h)//2:(max_h-h)//2+h, (max_w-w)//2:(max_w-w)//2+w] = img
        padded.append(pad)
    collated['img_L_array'] = (torch.stack(padded)
                               .permute(0, 3, 1, 2).float().div(255.0))
    return collated


def get_proj_lambda(global_step, args):
    if args.lambda_proj <= 0:
        return 0.0
    if global_step < args.proj_warmup_steps:
        return 0.0
    progress = (global_step - args.proj_warmup_steps) / max(
        args.max_train_steps - args.proj_warmup_steps, 1)
    return args.lambda_proj * min(progress * 2.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    args.tracker_project_name = os.path.join(
        "training_results", args.tracker_project_name)
    
    validate_args(args)

    logging_dir = Path(args.tracker_project_name, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.tracker_project_name, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.tracker_project_name, "checkpoints"), exist_ok=True)
        if not args.debug:
            wandb.init(project="CompMVR", name=args.tracker_project_name)

    global_step = 0
    resume_step=0
    if args.resume:
        ckpt_dir = os.path.join(args.tracker_project_name, "checkpoints")
        latest   = get_latest_checkpoint(ckpt_dir)
        if latest:
            ckpt_path, global_step = latest
            accelerator.load_state(ckpt_path)
            print(f"Resuming from step {global_step}")
            resume_step = global_step
    global_step = resume_step if args.resume else 0

    model_gen = CompMVR(args, num_views=args.num_views)

    model_gen.set_train()
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.gan_loss:
        model_reg = Discriminator(args=args, accelerator=accelerator)
        model_reg.set_train()

    loss_fn_dists = pyiqa.create_metric('dists', device=accelerator.device, as_loss=True)
    psnr_m    = pyiqa.create_metric('psnr',      device="cuda")
    msssim_m  = pyiqa.create_metric('ms_ssim',   device="cuda")
    lpips_m   = pyiqa.create_metric('lpips-vgg', device="cuda")
    dists_m   = pyiqa.create_metric('dists',     device="cuda")
    niqe_m    = pyiqa.create_metric('niqe',    device="cuda")
    musiq_m   = pyiqa.create_metric('musiq',     device="cuda")
    clipiqa_m = pyiqa.create_metric('clipiqa+',  device="cuda")

    lcorr_loss_fn = None
    if args.use_inter and args.use_lcorr:
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda")
        for param in vggt_model.parameters():
            param.requires_grad = False
        vggt_model.eval()
        
        lcorr_loss_fn = VGGTCorrespondenceLoss(
            max_pairs=args.num_views-1,         
            subsample=16
        )
        print(f"[L_corr] VGGTCorrespondenceLoss")
    

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    layers_to_opt = []
    for n, p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    for n, p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(p)
    layers_to_opt += list(model_gen.proj.parameters())
    if args.use_intra:
        layers_to_opt += list(model_gen.damage_estimator.parameters())
        if args.use_spatial_r and model_gen.r_spatial_adapter is not None:
            layers_to_opt += list(model_gen.r_spatial_adapter.parameters())
    if args.use_inter:
        layers_to_opt += model_gen.get_inter_params()
    print(f"[Opt] Intra(+inter): {sum(p.numel() for p in layers_to_opt):,} params")

    optimizer = torch.optim.AdamW(
        layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    if args.gan_loss:
        layers_to_opt_reg = []
        for n, p in model_reg.unet.named_parameters():
            if "lora" in n:
                layers_to_opt_reg.append(p)
        for p in model_reg.cls_pred_branch.parameters():
            layers_to_opt_reg.append(p)
        layers_to_opt_reg.append(model_reg.embeddings)
        optimizer_reg = torch.optim.AdamW(
            layers_to_opt_reg, lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
        lr_scheduler_reg = get_scheduler(
            args.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    comp = None
    if args.comp_path is not None:
        from cpe.cpe import CPE
        comp = CPE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
        comp.load_state_dict(torch.load(args.comp_path), strict=True)
        comp.eval()
        for p in comp.parameters():
            p.requires_grad = False
        comp = comp.to("cuda")

    # Dataset
    args.datasets = option.parse_dataset(args.datasets)['datasets']
    for phase, dataset_opt in args.datasets.items():
        if phase == 'train':
            if args.use_inter:
                from data.dataset import MultiViewDataset, multiview_collate_fn
                dataset_opt['num_views'] = args.num_views
                train_set = MultiViewDataset(dataset_opt, num_views=args.num_views)
                train_set.normalize = True
                scene_bs  = args.dataloader_batch_size
                dl_train  = DataLoader(
                    train_set, batch_size=scene_bs, shuffle=True,
                    num_workers=dataset_opt.get('dataloader_num_workers', 0),
                    drop_last=True, pin_memory=True,
                    collate_fn=multiview_collate_fn)
                print(f"[Data] MultiView: {len(train_set)} scenes, "
                      f"batch={scene_bs}×{args.num_views}={scene_bs*args.num_views} img/iter")
            else:
                from data.dataset import Dataset
                train_set = Dataset(dataset_opt)
                train_set.normalize = True
                dl_train  = DataLoader(
                    train_set,
                    batch_size=args.dataloader_batch_size,
                    shuffle=dataset_opt['dataloader_shuffle'],
                    num_workers=dataset_opt['dataloader_num_workers'],
                    drop_last=True, pin_memory=True,
                    collate_fn=custom_collate_fn)
        elif phase == 'test':
            from data.dataset import Dataset
            test_set    = Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                     num_workers=1, drop_last=False, pin_memory=True)

    # Accelerate prepare
    if args.gan_loss:
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, \
            lr_scheduler, lr_scheduler_reg = accelerator.prepare(
                model_gen, model_reg, optimizer, optimizer_reg,
                dl_train, lr_scheduler, lr_scheduler_reg)
    else:
        model_gen, optimizer, dl_train, lr_scheduler = accelerator.prepare(
            model_gen, optimizer, dl_train, lr_scheduler)

    if accelerator.is_main_process:
        del args.datasets
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=global_step,
            desc="Steps",
            disable=not accelerator.is_local_main_process,
            total=args.max_train_steps
        )   
   

    # ══════════════════════════════════════════════════════════════════
    # Training Loop
    # ══════════════════════════════════════════════════════════════════
    while True:
        for step, batch in enumerate(dl_train, start=global_step):
            global_step += 1
            if global_step > args.max_train_steps:
                exit()

            m_acc = [model_gen, model_reg] if args.gan_loss else [model_gen]

            with accelerator.accumulate(*m_acc):

                if args.use_inter:
                    images_L = batch["images_L"].to("cuda")  # (B, V, 3, H, W)
                    images_H = batch["images_H"].to("cuda")
                    B_scene, V = images_L.shape[0], images_L.shape[1]
                    H_, W_     = images_L.shape[3], images_L.shape[4]

                    x_src = images_L.reshape(B_scene * V, 3, H_, W_)
                    x_tgt = images_H.reshape(B_scene * V, 3, H_, W_)

                    img_L_arr     = batch["img_L_array"].to("cuda")
                    x_src_comp_01 = (img_L_arr.reshape(B_scene*V, 3, H_, W_)
                                     * 0.5 + 0.5).clamp(0, 1)
                else:
                    x_src         = batch["L"].to("cuda")
                    x_tgt         = batch["H"].to("cuda")
                    
                    x_src_comp_01 = batch["img_L_array"]

                with torch.no_grad():
                    visual_embedding = comp.get_visual_embedding(x_src_comp_01) if comp is not None else None

                out = model_gen(x_src, visual_embedding, c_tgt=x_tgt)

                x_pred = (out["output_image"] * 0.5 + 0.5).clamp(0, 1) # [0,1]
                x_tgt_ = (x_tgt * 0.5 + 0.5).clamp(0, 1) # [0,1]


                l2_loss    = F.mse_loss(x_pred.float(), x_tgt_.float())
                dists_loss = loss_fn_dists(x_pred.float(), x_tgt_.float())
                loss       = l2_loss + dists_loss

                r_loss = torch.tensor(0.0, device="cuda")
                if args.use_intra and out["r_gt"] is not None:
                    r_loss = compression_damage_loss(
                        out["r_pred"], out["r_gt"], alpha=args.r_loss_alpha)
                    loss = loss + r_loss * args.r_loss_weight
                    
                    if args.resi_pred_vis and accelerator.is_main_process:
                        vis_dir = os.path.join(args.tracker_project_name, 
                            f"r_gt_vis/step_{global_step}")
                        visualize_residual(out["r_pred"].detach(), save_path=os.path.join(vis_dir, f"r_pred_{batch['frame_id']}.png"))
                        visualize_residual(out["r_gt"].detach(), save_path=os.path.join(vis_dir, f"r_gt_{batch['frame_id']}.png"))

                # vggt loss
                L_corr = torch.tensor(0.0, device="cuda")
                if args.use_inter and lcorr_loss_fn is not None:
                    restored_01 = (out["output_image"] * 0.5 + 0.5).clamp(0, 1)
                    hq_01 = (x_tgt * 0.5 + 0.5).clamp(0, 1)

                    restored_list = restored_01.view(B_scene, V, 3, H_, W_).unbind(dim=1)
                    hq_list       = hq_01.view(B_scene, V, 3, H_, W_).unbind(dim=1)
                    L_corr = lcorr_loss_fn(
                            restored_imgs=restored_list, 
                            hq_imgs=hq_list,              
                            vggt_model=vggt_model,
                        )
                    loss = loss + args.lambda_lcorr * L_corr

                # GAN
                if args.gan_loss:
                    if torch.cuda.device_count() > 1:
                        gen_loss = model_reg.module.compute_generator_loss(out["model_pred"])
                    else:
                        gen_loss = model_reg.compute_generator_loss(out["model_pred"])
                    loss = loss + gen_loss * args.gan_gen_weight

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if args.gan_loss:
                    x_tgt_gan = x_tgt * 2 - 1
                    if torch.cuda.device_count() > 1:
                        gt_latents = model_reg.module.compute_gt_latents(x_tgt_gan)
                        loss_d     = model_reg.module.compute_discriminator_loss(
                            gt_latents, out["model_pred"]) * args.gan_dis_weight
                    else:
                        gt_latents = model_reg.compute_gt_latents(x_tgt_gan)
                        loss_d     = model_reg.compute_discriminator_loss(
                            gt_latents, out["model_pred"]) * args.gan_dis_weight
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if accelerator.is_main_process:
                    logs = {
                        "loss_l2":       l2_loss.detach().item(),
                        "loss_dists":    dists_loss.detach().item(),
                        "loss_r_pred":   (r_loss * args.r_loss_weight).detach().item(),
                        "loss_lcorr":    (args.lambda_lcorr * L_corr).detach().item(),
                    }
                    if args.gan_loss:
                        logs["loss_g"] = gen_loss.detach().item()
                        logs["loss_d"] = loss_d.detach().item()
                    progress_bar.set_postfix(**logs)
                    if not args.debug:
                        wandb.log(logs)
                    accelerator.log(logs, step=global_step)

            # Checkpoint & Validation
            save_dir = os.path.join(args.tracker_project_name, "checkpoints")
            if (global_step % args.checkpointing_steps == 0
                    or global_step == args.max_train_steps):

                outf = os.path.join(save_dir, f"model_{global_step}.pkl")
                if global_step == args.max_train_steps:
                    gen_unwrap = (accelerator.unwrap_model(model_gen)
                                  if torch.cuda.device_count() > 1 else model_gen)
                    gen_unwrap.save_model(outf)
                else:
                    accelerator.save_state(outf)

                # Validation

                gen = (accelerator.unwrap_model(model_gen)
                       if torch.cuda.device_count() > 1 else model_gen)
                gen.set_eval()

                n_total = len(test_set)
                val_indices = (random.sample(range(n_total), args.val_num_samples)
                               if args.val_num_samples > 0 and n_total > args.val_num_samples
                               else list(range(n_total)))
                val_loader = DataLoader(
                    torch.utils.data.Subset(test_set, val_indices),
                    batch_size=1, shuffle=False, num_workers=1,
                    drop_last=False, pin_memory=True)

                test_results = {k: [] for k in
                                ['psnr', 'msssim', 'lpips', 'dists', 'niqe', 'musiq', 'clipiqa']}

                for test_data in val_loader:
                    img_name, _ = os.path.splitext(
                        os.path.basename(test_data['H_path'][0]))

                    img_H = Image.open(test_data['H_path'][0]).convert('RGB')
                    img_L = Image.open(test_data['L_path'][0]).convert('RGB')
                    new_w = img_L.width  - img_L.width  % 8
                    new_h = img_L.height - img_L.height % 8
                    img_L = img_L.resize((new_w, new_h), Image.LANCZOS)
                    img_H = img_H.resize((new_w, new_h), Image.LANCZOS)

                    img_L_np     = np.array(img_L)
                    img_L_tensor = (torch.from_numpy(img_L_np)
                                    .permute(2,0,1).float().div(255.)
                                    .unsqueeze(0).to("cuda"))

                    with torch.no_grad():
                        visual_embedding = comp.get_visual_embedding(img_L_tensor) if comp is not None else None

                    lq = tensor_transforms(img_L).unsqueeze(0).to("cuda")
                    with torch.no_grad():
                        lq    = lq * 2 - 1
                        img_E = gen.eval(lq, visual_embedding)
                        img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
                        if args.align_method == 'adain':
                            img_E = adain_color_fix(target=img_E, source=img_L)
                        elif args.align_method == 'wavelet':
                            img_E = wavelet_color_fix(target=img_E, source=img_L)

                    img_H_np = np.array(img_H)
                    img_E_np = np.array(img_E)
                    util.imsave(img_E_np, os.path.join(
                        args.tracker_project_name, img_name + '.png'))

                    img_E_t = (torch.tensor(img_E_np/255., device="cuda")
                               .permute(2,0,1).unsqueeze(0).float())
                    img_H_t = (torch.tensor(img_H_np/255., device="cuda")
                               .permute(2,0,1).unsqueeze(0).float())


                    test_results['psnr'].append(psnr_m(img_E_t, img_H_t).item())
                    test_results['msssim'].append(msssim_m(img_E_t, img_H_t))
                    test_results['lpips'].append(lpips_m(img_E_t, img_H_t).item())
                    test_results['dists'].append(dists_m(img_E_t, img_H_t).item())
                    test_results['niqe'].append(niqe_m(img_E_t).item())
                    test_results['musiq'].append(musiq_m(img_E_t, img_H_t).item())
                    test_results['clipiqa'].append(clipiqa_m(img_E_t, img_H_t).item())

                avg = {k: sum(v)/len(v) for k, v in test_results.items()}
                if accelerator.is_main_process and not args.debug:
                    wandb.log({'PSNR':    avg['psnr'],   'MSSSIM':  avg['msssim'],
                               'LPIPS':   avg['lpips'],  'DISTS':   avg['dists'],
                               'NIQE':    avg['niqe'],   'MUSIQ':   avg['musiq'],
                               'CLIPIQA': avg['clipiqa']})

                gen.set_train()

            accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = parse_args()
    main(args)

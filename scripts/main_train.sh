#!/bin/bash
set -e 
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

COMMON_ARGS="
    --pretrained_model=./model_zoo/stable-diffusion-2-1-base \
    --learning_rate=5e-5 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps 6000 \
    --mixed_precision=fp16 \
    --report_to "tensorboard" \
    --seed 123 \
    --lora_rank=16 \
    --max_train_steps=100000 \
    --timestep 499 \
    --resume \
    --datasets ./options/config.json
"
LAUNCH="accelerate launch --main_process_port 18888 ./scripts/main_train.py"


$LAUNCH $COMMON_ARGS \
    --comp_path ./checkpoints/final_stage1.pth \
    --tracker_project_name "train_CompMVR" \
    --use_intra --use_inter --num_views 4 --use_spatial_r --use_lcorr --lambda_lcorr 0.01 --dataloader_batch_size 1


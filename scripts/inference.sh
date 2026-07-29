export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

COMMON_ARGS="
    --pretrained_model=./pretrained/stable-diffusion-2-1-base \
    --timestep 499 \
    --seed 123 \
    --lora_rank=16 \
    --testset_root ./dataset/testset/ \
"
LAUNCH="python ./scripts/inference.py"

$LAUNCH $COMMON_ARGS \
    --comp_path ./checkpoints/final_stage1.pth \
    --compmvr_path ./checkpoints/main_train_residual_UNet_input.pkl \
    --save_root "results/ablation/main_train_residual_UNet_input" \
    --use_intra --use_inter --num_views 4 --use_spatial_r

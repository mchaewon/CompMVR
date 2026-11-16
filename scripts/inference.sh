export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# MODE=restore : restore every frame in INPUT_DIR (no GT, no metrics)
# MODE=eval    : evaluate on the structured testset (needs GT, computes metrics)
MODE=${MODE:-restore}

COMMON_ARGS="
    --pretrained_model=./model_zoo/stable-diffusion-2-1-base \
    --comp_path ./checkpoints/CPE.pth \
    --compmvr_path ./checkpoints/CompMVR.pkl \
    --timestep 499 \
    --seed 123 \
    --lora_rank=16 \
    --use_intra --use_inter --num_views 4 --use_spatial_r
"
LAUNCH="python ./scripts/inference.py"

if [ "$MODE" = "eval" ]; then
    $LAUNCH $COMMON_ARGS --eval \
        --testset_root ./dataset/testset/ \
        --gt_root ./dataset/testset/GT \
        --datasets Free \
        --save_root "results/eval"
else
    $LAUNCH $COMMON_ARGS \
        --input_dir ./dataset/my_frames \
        --save_root "results/restored"
fi

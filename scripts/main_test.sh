export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

COMMON_ARGS="
    --pretrained_model=./model_zoo/stable-diffusion-2-1-base \
    --timestep 499 \
    --seed 123 \
    --lora_rank=16 \
    --datasets ./options/config.json
"
LAUNCH="python ./scripts/main_test.py"

echo "===== [0/8] main_train_residual_UNet_input =====" 
$LAUNCH $COMMON_ARGS \
    --comp_path /path/comp_path \
    --ours_main_path train_CompMVR/checkpoints/100000.pkl \
    --save_root "results/" \
    --use_intra --use_inter --num_views 4 --use_spatial_r
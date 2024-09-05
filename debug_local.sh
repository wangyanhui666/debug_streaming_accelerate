#!/bin/bash
export WANDB_API_KEY="local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6"
wandb login local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6 --relogin --host=https://microsoft-research.wandb.io
accelerate launch \
    --num_processes 1 \
    --mixed_precision=bf16 \
    train_diffusers_imagenet.py \
    --dataset-path ~/t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256_debug \
    --use_ema \
    --resolution 32 \
    --train_batch_size=64 \
    --dataloader_num_workers 4 \
    --num_epochs 1000 \
    --save_model_epochs 1 \
    --output_dir ~/guangtingsc/t2vg/dit/logs/debug/0801_debug_256_speed_1 \
    --logger wandb \
    --mixed_precision bf16 \
    --checkpointing_steps 10 \

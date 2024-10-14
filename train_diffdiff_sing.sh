#!/bin/bash
export WANDB_API_KEY="local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6"
# --gradient_accumulation_steps=4 \
# --gradient_checkpointing \
wandb login local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6 --relogin --host=https://microsoft-research.wandb.io
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision=bf16 \
    train_diffdiff_imagenet.py \
    --dataset-path ~/t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256 \
    --use_ema \
    --resolution 32 \
    --train_batch_size=8 \
    --dataloader_num_workers 4 \
    --num_epochs 1000 \
    --save_model_epochs 1 \
    --output_dir ~/guangtingsc/t2vg/dit/logs/debug_imagenet_256/0918_diffdiff_256_1000_1000_debugmemory \
    --logger wandb \
    --mixed_precision bf16 \
    --checkpointing_steps 10000 \
    --resume_from_checkpoint latest \

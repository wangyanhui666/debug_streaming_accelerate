#!/bin/bash
export NCCL_SOCKET_IFNAME=eth0
export WANDB_API_KEY="local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6"
# --gradient_accumulation_steps=4 \
# --gradient_checkpointing \
wandb login local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6 --relogin --host=https://microsoft-research.wandb.io
accelerate launch \
    --config_file /home/t2vg-a100-G4-40/code/github/DiT/config/default_config.yaml \
    train_diffusers_imagenet.py \
    --local-dataset-path ~/datasets/vae_mds \
    --use_ema \
    --resolution 32 \
    --train_batch_size=64 \
    --dataloader_num_workers 4 \
    --num_epochs 1000 \
    --save_model_epochs 1 \
    --output_dir ~/guangtingsc/t2vg/dit/logs/train_imagenet_256/1009_imagenet_int8_test \
    --logger wandb \
    --mixed_precision bf16 \
    --checkpointing_steps 10000 \
    --resume_from_checkpoint latest \

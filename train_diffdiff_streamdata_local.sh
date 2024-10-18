#!/bin/bash
export WANDB_API_KEY="local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6"
# --gradient_accumulation_steps=4 \
# --gradient_checkpointing \
wandb login local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6 --relogin --host=https://microsoft-research.wandb.io
accelerate launch \
    --config_file /home/t2vg-a100-G4-42/code/DiT/config/default_config.yaml \
    train_diffdiff_imagenet_streamdata.py \
    --local-dataset-path ~/datasets/vae_mds_fp32 \
    --vae-path stabilityai/sd-vae-ft-mse \
    --use_ema \
    --resolution 32 \
    --train_batch_size=64 \
    --dataloader_num_workers 1 \
    --num_epochs 1000 \
    --save_model_epochs 40000 \
    --output_dir ~/guangtingsc/t2vg/dit/logs/train_imagenet_256/1018_diffdiff_imagenet_fp32_1000_1000_test1 \
    --logger wandb \
    --mixed_precision bf16 \
    --checkpointing_steps 40000 \
    --resume_from_checkpoint latest \
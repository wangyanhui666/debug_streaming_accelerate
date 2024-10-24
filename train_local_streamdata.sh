#!/bin/bash
export NCCL_SOCKET_IFNAME=eth0
# export WANDB_API_KEY="your_wandb_api_key"
# --gradient_accumulation_steps=4 \
# --gradient_checkpointing \
# export WANDB_API_KEY="your_wandb_api_key"
# # wandb login
# wandb login

accelerate launch \
    --config_file ./config/default_config.yaml \
    train_diffusers_imagenet_streamdata.py \
    --local-dataset-path ./vae_mds_fp32 \
    --vae-path stabilityai/sd-vae-ft-mse \
    --use_ema \
    --resolution 32 \
    --train_batch_size=64 \
    --dataloader_num_workers 1 \
    --num_epochs 1000 \
    --save_model_epochs 40000 \
    --output_dir ./logs \
    --logger tensorboard \
    --mixed_precision bf16 \
    --checkpointing_steps 40000 \
    --resume_from_checkpoint latest \

#!/bin/bash

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    train.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --log-every 1 \
    --features-path ~/t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256_debug \
    --results-dir ~/guangtingsc/t2vg/dit/logs/debug/0801_debug_256_speed_1
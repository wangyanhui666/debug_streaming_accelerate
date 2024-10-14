torchrun --nnodes=1 \
    --nproc_per_node=4 \
    sample_dit_ddp.py \
    --dtype bf16 \
    --vae-path stabilityai/sdxl-vae \
    --ckpt ~/guangtingsc/t2vg/dit/logs/train_imagenet_256/1011_imagenet_int8_fix_vae/checkpoint-600000 \
    --num-fid-samples 50000 \
    --global-seed 1 \
    --use-ema
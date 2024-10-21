torchrun --nnodes=1 \
    --nproc_per_node=4 \
    sample_dit_ddp.py \
    --dtype fp32 \
    --vae-path stabilityai/sd-vae-ft-mse \
    --ckpt ~/guangtingsc/t2vg/dit/logs/train_imagenet_256/1018_imagenet_fp32_fix_flip_writedatabug_2/checkpoint-400000 \
    --num-fid-samples 50000 \
    --global-seed 1 \
    --use-ema
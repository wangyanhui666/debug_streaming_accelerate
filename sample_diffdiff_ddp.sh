torchrun --nnodes=1 \
    --nproc_per_node=4 \
    sample_diffdiff_ddp.py \
    --dtype bf16 \
    --vae-path stabilityai/sd-vae-ft-mse \
    --ckpt ~/guangtingsc/t2vg/dit/logs/train_imagenet_256/1018_diffdiff_imagenet_fp32_1000_1000_test1/checkpoint-400000 \
    --num-sampling-steps1 2 \
    --num-sampling-steps2 50 \
    --num-fid-samples 50000 \
    --global-seed 1 \
    --use-ema
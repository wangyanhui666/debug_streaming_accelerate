torchrun --nnodes=1 \
    --nproc_per_node=4 \
    sample_dit_ddp.py \
    --ckpt ~/guangtingsc/t2vg/dit/logs/train_imagenet_256/0905_dit_256_test_memory_1/checkpoint-400000 \
    --num-fid-samples 50000 \
    --global-seed 1 \
    --no-use-ema 
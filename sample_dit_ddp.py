# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from diffusion_diffusers.model import DiT_XL_2
from download import find_model
from diffusion import create_diffusion
from diffusers import DDPMPipeline, DiTPipeline, DDPMScheduler, AutoencoderKL,DiTTransformer2DModel
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    assert args.ckpt is not None, "Please provide a path to a DiT checkpoint via --ckpt"

    # Load model:
    latent_size = args.image_size // 8
    if args.use_ema:
        model=DiTTransformer2DModel.from_pretrained(args.ckpt, subfolder="transformer_ema")
    else:
        model=DiTTransformer2DModel.from_pretrained(args.ckpt, subfolder="transformer")
    model.to(device)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    noise_scheduler = DDPMScheduler(clip_sample=False)
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    pipeline = DiTPipeline(model, vae, noise_scheduler).to(dtype=dtype)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    checkpoint_name = os.path.basename(os.path.normpath(args.ckpt))
    parent_dir = os.path.dirname(os.path.normpath(args.ckpt))
    inference_folder_name = f'inference_{checkpoint_name}-ema-{args.use_ema}-image_size-{args.image_size}-cfg-{args.cfg_scale}-dtype-{args.dtype}-seed-{args.global_seed}'
    sample_folder_dir = os.path.join(parent_dir, inference_folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    samples_list = []
    for _ in pbar:
        # Sample inputs:
        
        y = torch.randint(0, args.num_classes, (n,), device=device)
        y = y.tolist()
        images = pipeline(
            class_labels=y,
            guidance_scale=1.0,
            num_inference_steps=250,
            ).images
        
        # Save samples to disk as individual .png files and collect sample data
        for i, sample in enumerate(images):
            index = i * dist.get_world_size() + rank + total
            sample.save(f"{sample_folder_dir}/{index:06d}.png")
            sample_np = np.asarray(sample).astype(np.uint8)
            samples_list.append(sample_np)
        total += global_batch_size

    # 将本进程的样本保存为 .npz 文件
    samples_np = np.stack(samples_list)
    npz_path = f"{sample_folder_dir}/samples_rank_{rank}.npz"
    np.savez(npz_path, arr_0=samples_np)
    print(f"Rank {rank}: Saved .npz file to {npz_path} [shape={samples_np.shape}].")
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    # if rank == 0:
    #     create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
    #     print("Done.")
    # dist.barrier()
    # 主进程合并所有 .npz 文件
    if rank == 0:
        print("Merging .npz files from all ranks...")
        all_samples = []
        for r in range(dist.get_world_size()):
            npz_path = f"{sample_folder_dir}/samples_rank_{r}.npz"
            data = np.load(npz_path)['arr_0']
            all_samples.append(data)
        all_samples = np.concatenate(all_samples, axis=0)
        # 只保留所需数量的样本
        all_samples = all_samples[:args.num_fid_samples]
        npz_path = f"{sample_folder_dir}.npz"
        np.savez(npz_path, arr_0=all_samples)
        print(f"Saved merged .npz file to {npz_path} [shape={all_samples.shape}].")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-ema", help="Path to the VAE model.")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")  
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

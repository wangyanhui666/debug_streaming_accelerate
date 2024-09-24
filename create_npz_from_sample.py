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

    create_npz_from_sample_folder(args.sample_folder_dir, args.num_fid_samples)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--sample-folder-dir", type=str, default="/home/t2vg-a100-G4-40/guangtingsc/t2vg/dit/logs/train_imagenet_256/0905_dit_256_test_memory_1/inference_checkpoint-400000-image_size-256-cfg-1.0-seed-0")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
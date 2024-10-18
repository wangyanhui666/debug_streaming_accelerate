import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from streaming import MDSWriter

import logging
import time
import numpy as np
from typing import Any
from datasets import load_dataset

import torch.distributed as dist

# Initialize logging
logging.basicConfig(level=logging.INFO)

from preprocess_data.imagenet_labels import IMGNET_LABELS
from streaming.base.format.mds.encodings import Encoding, _encodings


# class uint8(Encoding):
#     def encode(self, obj: Any) -> bytes:
#         return obj.tobytes()

#     def decode(self, data: bytes) -> Any:
#         return np.frombuffer(data, np.uint8)


class np32(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float32)


_encodings["np32"] = np32
# _encodings["uint8"] = uint8


def center_crop_imagenet(image_size: int, pil_image: Image) -> np.ndarray:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    assert len(new_size) == 2
    pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def prepare_image(arr):
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


class SubImageDataset(Dataset):
    def __init__(self, idx_range=(0, 1000), args=None):

        self.dataset = load_dataset("imagenet-1k", split="train",trust_remote_code=True)
        self.idx_range = idx_range
        self.image_size = args.image_size
        # modify the idx range if its out of bounds
        if self.idx_range[1] > len(self.dataset):
            self.idx_range = (self.idx_range[0], len(self.dataset))

    def __len__(self):
        return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        idx = idx + self.idx_range[0]
        image, label = self.dataset[idx]["image"], self.dataset[idx]["label"]
        image = image.convert("RGB")
        # image is pil. first center-crop to 256x256
        # for that, first resize to 256 x N or N x 256, then crop to 256 x 256
        image = center_crop_imagenet(self.image_size, image)
        
        image = prepare_image(image)

        return image, label


from tqdm import tqdm
from torch.utils.data import DataLoader


@torch.no_grad()
def convert_to_mds(
    idx_range, out_root, device, batch_size=8, num_workers=4, args=None
):
    logging.info(f"Processing on {device}")

    # Load the VAE model
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae_model = vae_model.to(device).eval()


    # vae_model.to(memory_format=torch.channels_last)
    # vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead", fullgraph=False)

    # Create the dataset and dataloader
    dataset = SubImageDataset(idx_range, args)
    
    if dataset.__len__() < 1:
        logging.info("No images to process.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )


    columns = {"vae_output": "np32", "label": "int", "label_as_text" : 'str'}

    if os.path.exists(out_root):
        # remove all files in the directory
        for f in os.listdir(out_root):
            os.remove(os.path.join(out_root, f))
    os.makedirs(out_root, exist_ok=True)

    with MDSWriter(out=out_root, columns=columns) as out:
        inference_latencies = []

        for batch in tqdm(dataloader):
            start_time = time.time()

            processed_images, labels = batch
            # save the flipped images as well! important for training
            processed_images= torch.cat([processed_images, processed_images.flip(dims=[-1])], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            processed_images = processed_images.to(device)
            vae_outputs = vae_model.encode(processed_images).latent_dist.sample()
            

            # Iterate through the batch
            for i in range(len(labels)):
                sample = {
                    "vae_output": vae_outputs[i].cpu().numpy().astype(np.float32),
                    "label": labels[i].item(),
                    "label_as_text": IMGNET_LABELS[labels[i].item()],
                }
                out.write(sample)

            inference_latencies.append(time.time() - start_time)

        logging.info(
            f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
        )


def distribute_indices_to_gpus(total_indices, num_gpus):
    # 计算每个 GPU 处理的基本索引数
    base_size = total_indices // num_gpus
    remainder = total_indices % num_gpus  # 计算剩余的索引数

    indices_ranges = []

    start_index = 0
    for i in range(num_gpus):
        # 如果是最后一个 GPU，处理剩余所有数据
        if i == num_gpus - 1:
            end_index = total_indices
        else:
            end_index = start_index + base_size
            # 给一些 GPU 分配多余的索引（尽量均匀）
            if remainder > 0:
                end_index += 1
                remainder -= 1
        
        # 将当前 GPU 负责的索引范围加入到列表
        indices_ranges.append((start_index, end_index - 1))
        
        # 更新下一个区间的起始索引
        start_index = end_index

    return indices_ranges



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to MDS format.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing (cuda or cpu).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility."
    )
    parser.add_argument("--image-size", type=int, default=256, help="Image size.")
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )
    parser.add_argument(
        "--out_root", type=str, default="./vae_mds", help="Output root directory."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    args = parser.parse_args()


    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={args.seed}, world_size={world_size}.")

    # imagenet has 1281167 images
    total_num_images = 1281167
    this_gpu_indices_ranges = distribute_indices_to_gpus(total_num_images, world_size)[rank]
    out_path=os.path.join(args.out_root, f"{str(rank)}")
    print(f"Processing on {device}")
    print("this_gpu_indices_ranges", this_gpu_indices_ranges)
    convert_to_mds(this_gpu_indices_ranges, out_path, device, args.batch_size, 8, args=args)
    logging.info("Finished processing images.")
import streaming
from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np
from typing import Any
import torch
from torch.utils.data import DataLoader
from streaming import StreamingDataset
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x=  np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 28

_encodings["uint8"] = uint8

remote_train_dir = "/home/t2vg-a100-G4-40/datasets/vae_mds" # this is the path you installed this dataset.
local_train_dir = "/home/t2vg-a100-G4-40/code/github/DiT/results/local_train_dir"
streaming.base.util.clean_stale_shared_memory()
train_dataset = StreamingDataset(
    local=local_train_dir,
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size = 32
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=3,
)


vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda:0")
batch = next(iter(train_dataloader))
# i = 0
# vae_latent = batch["vae_output"].reshape(-1, 4, 32, 32)[i:i+1].cuda().float()
# idx = batch["label"][i]
# text_label = batch['label_as_text'][i]

# print(f"idx: {idx}, text_label: {text_label}, latent: {vae_latent.shape}")
# # idx: 402, text_label: acoustic guitar, latent: torch.Size([1, 4, 32, 32])

# # example decoding
# x = vae.decode(vae_latent.cuda()).sample
# img = VaeImageProcessor().postprocess(image = x.detach(), do_denormalize = [True, True])[0]
# img.save("5th_image.png")
# 计算方差
latents_sum = 0
latents_square_sum = 0
total_samples = 0

# 遍历数据集，累加每个 batch 的 latent
for batch in tqdm(iter(train_dataloader), desc="Processing batches"):
    latents = batch["vae_output"].reshape(-1, 4, 32, 32).float().cuda()
    latents_sum += torch.sum(latents)
    latents_square_sum += torch.sum(latents ** 2)
    total_samples += latents.numel()

# 计算均值和方差
latents_mean = latents_sum / total_samples
latents_variance = (latents_square_sum / total_samples) - (latents_mean ** 2)

# 打印方差
print(f"Latent Variance: {latents_variance}")
import argparse
import os
import torch
from PIL import Image
from datetime import timedelta

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDPMScheduler,DDIMScheduler, Transformer2DModel, DiTTransformer2DModel
from diffusion_diffusers.dataset import CustomDataset

from diffusion_diffusers.model import DiT_XL_2

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset-path", type=str, default="/home/t2vg-a100-G4-40/t2vgusw2/videos/imagenet/sd_latents/dit_extact_latents_256_debug", help="Path to the dataset")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sd-vae-ft-ema", help="Path to the VAE model.")
    parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return args
if __name__=="__main__":
    # args = parse_args()
    # # logging_dir = os.path.join(args.output_dir, args.logging_dir)
    # # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with="wandb",
    #     # project_config=accelerator_project_config,
    #     kwargs_handlers=[kwargs],
    # )
    # features_dir = f"{args.dataset_path}/imagenet512_features"
    # labels_dir = f"{args.dataset_path}/imagenet512_labels"
    # dataset = CustomDataset(features_dir, labels_dir)
    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=True, num_workers=1
    # )
    # vae = AutoencoderKL.from_pretrained(args.vae_path)
    # vae.requires_grad_(False)
    # vae.to(accelerator.device, dtype=torch.float32)
    # train_dataloader=accelerator.prepare(
    #     train_dataloader
    # )
    # test_dir = os.path.join("./results/clean_image")
    # os.makedirs(test_dir,exist_ok=True)
    # for step, batch in enumerate(train_dataloader):
    #     latents=batch["features"].to(accelerator.device)
    #     latents=latents.squeeze(dim=1)
    #     labels=batch["labels"].to(accelerator.device)
    #     labels=labels.squeeze(dim=1)
    #     latents = 1 / vae.config.scaling_factor * latents
    #     samples = vae.decode(latents).sample
    #     samples = (samples / 2 + 0.5).clamp(0, 1)
    #     samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
    #     samples = numpy_to_pil(samples)
    #     for i in range(len(samples)):
    #         samples[i].save(f"{test_dir}/{step}_{i}.png")
    torch.manual_seed(33)
    scheduler_config = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256").scheduler.config
    # noise_scheduler = DDPMScheduler.from_config(scheduler_config)
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler(clip_sample=False)
    pipe=pipe.to("cuda")
    device = "cuda"
    # model=Transformer2DModel.from_pretrained("facebook/DiT-XL-2-256", subfolder="transformer")
    # model=DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256", subfolder="transformer")
    vae = AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256",subfolder="vae")
    model=DiTTransformer2DModel.from_pretrained("/home/t2vg-a100-G4-40/guangtingsc/t2vg/dit/logs/train_imagenet_256/0905_dit_256_test_memory_1/checkpoint-400000",subfolder="transformer_ema")
    print(model.config)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    # pipe = DiTPipeline(model, vae, noise_scheduler).to(torch.bfloat16)
    pipe.to(device)
    generator=torch.Generator(device=device).manual_seed(33)
    output = pipe(class_labels=[0,0,0,0], num_inference_steps=50, guidance_scale=1.0, generator=generator)
    save_dir = "./results/official_dit_samples_usemy_scheduler_float16"
    os.makedirs(save_dir,exist_ok=True)
    for i in range(len(output.images)):
        output.images[i].save(f"{save_dir}/{i}.png")
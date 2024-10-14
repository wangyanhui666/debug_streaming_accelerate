import os
from diffusion_diffusers.pipeline import DiffDiffPipeline
from diffusers import DDPMScheduler, DDIMScheduler, DiTPipeline,AutoencoderKL,DiTTransformer2DModel
import torch
import time
from dataclasses import dataclass
from diffusers.utils import make_image_grid


def generate_images_for_steps(pipeline, config, steps_pairs, save_dir="samples", rows=4, cols=4):
    class_labels = [i for i in range(config.eval_batch_size)]

    # 创建保存结果的目录
    test_dir = os.path.join(config.output_dir, save_dir)
    os.makedirs(test_dir, exist_ok=True)

    for step1, step2 in steps_pairs:
        generator = torch.Generator("cuda").manual_seed(123)
        start_time = time.time()

        images = pipeline(
            class_labels=class_labels,
            guidance_scale1=1.0,
            guidance_scale2=1.0,
            num_inference_steps1=step1,
            num_inference_steps2=step2,
            generator=generator,  # 使用独立的torch generator避免影响主训练循环的随机状态
        ).images

        # 计算生成图像所花费的时间（毫秒）
        elapsed_time = int((time.time() - start_time) * 1000)

        # 将图像制作成网格
        image_grid = make_image_grid(images, rows=rows, cols=cols)

        # 保存图像，文件名中包含step1, step2和生成时间
        image_filename = f"{step1}_{step2}_{elapsed_time}ms.png"
        image_grid.save(os.path.join(test_dir, image_filename))

def generate_images_for_steps_baseline(pipeline, config, steps_pairs, save_dir="samples", rows=4, cols=4):
    class_labels = [i for i in range(config.eval_batch_size)]

    # 创建保存结果的目录
    test_dir = os.path.join(config.output_dir, save_dir)
    os.makedirs(test_dir, exist_ok=True)

    for step1, step2 in steps_pairs:
        generator = torch.Generator("cuda").manual_seed(123)
        start_time = time.time()
        images = pipeline(
            class_labels=class_labels,
            guidance_scale=1.0,
            num_inference_steps=step1,
            generator=generator,  # 使用独立的torch generator避免影响主训练循环的随机状态
        ).images

        # 计算生成图像所花费的时间（毫秒）
        elapsed_time = int((time.time() - start_time) * 1000)

        # 将图像制作成网格
        image_grid = make_image_grid(images, rows=rows, cols=cols)

        # 保存图像，文件名中包含step1, step2和生成时间
        image_filename = f"{step1}_{elapsed_time}ms.png"
        image_grid.save(os.path.join(test_dir, image_filename))

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    patch_size = 2
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 600000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 300
    save_model_epochs = 1000
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/1011_imagenet_int8_fix_vae/inference"  # the model name locally and on the HF Hub
    
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "wangyanhui666/test_diffdiff3"  # the name of the repository to create on the HF Hub
    hub_private_repo = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()
# Load the pipeline
transforms=DiTTransformer2DModel.from_pretrained("/home/t2vg-a100-G4-40/guangtingsc/t2vg/dit/logs/train_imagenet_256/1011_imagenet_int8_fix_vae/checkpoint-600000", subfolder="transformer_ema")
transforms.eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae.to("cuda",dtype=torch.float32)
noise_scheduler=DDPMScheduler(clip_sample=False)
pipeline=DiTPipeline(transformer=transforms, vae=vae, scheduler=noise_scheduler)
pipeline=pipeline.to(dtype=torch.float32)
# pipeline = DiTPipeline.from_pretrained("/home/t2vg-a100-G4-40/guangtingsc/t2vg/dit/logs/train_imagenet_256/1011_imagenet_int8_fix_vae",torch_dtype=torch.float32, use_safetensors=True)
# pipeline.scheduler.clip_sample=False

# steps_pairs = [(1, 20), (2, 20), (3, 20), (4, 20), (5, 20), (10, 20), (20,20),(30,20),(40,20),(50,20)]  # 定义不同的step1, step2组合
steps_pairs=[(50,50)]
# generate_images_for_steps(pipeline, config, steps_pairs)
if isinstance(pipeline, DiTPipeline):
    # scheduler=DDPMScheduler.from_config(pipeline.scheduler.config)
    # # scheduler.clip_sample=False
    # pipeline.scheduler = scheduler
    pipeline = pipeline.to("cuda")
    generate_images_for_steps_baseline(pipeline, config, steps_pairs)
else:
    scheduler1=DDIMScheduler.from_config(pipeline.scheduler1.config)
    scheduler2=DDIMScheduler.from_config(pipeline.scheduler2.config)
    pipeline.scheduler1 = scheduler1
    pipeline.scheduler2 = scheduler2
    pipeline = pipeline.to("cuda")
    generate_images_for_steps(pipeline, config, steps_pairs)


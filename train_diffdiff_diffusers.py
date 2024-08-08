import threading
from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import DDPMScheduler,DiTPipeline
from diffusion_diffusers.model import DiT_XL_2, Vae, DiT_S_2
from diffusion_diffusers.pipeline import DiffDiffPipeline
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator,notebook_launcher
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from huggingface_hub import notebook_login

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    class_labels=[0 for i in range(config.eval_batch_size)]
    images = pipeline(
        class_labels=class_labels,
        guidance_scale1=1.0,
        guidance_scale2=1.0,
        num_inference_steps1=50,
        num_inference_steps2=50,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}_{50}_{50}.png")

    images = pipeline(
        class_labels=class_labels,
        guidance_scale1=1.0,
        guidance_scale2=1.0,
        num_inference_steps1=1,
        num_inference_steps2=50,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}_{1}_{50}.png")

def save_model(config, repo_id, epoch,pipeline):
    if False:
        upload_folder(
            repo_id=repo_id,
            folder_path=config.output_dir,
            commit_message=f"Epoch {epoch}",
            ignore_patterns=["step_*", "epoch_*"],
        )
    else:
        pipeline.save_pretrained(config.output_dir)

def train_loop(config, model, model2, vae, noise_scheduler1,noise_scheduler2, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    device = accelerator.device
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, model2, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, model2, vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            noise2 = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler1.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            timesteps2 = torch.randint(
                0, noise_scheduler2.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            # TODO need to chage to real class label
            class_labels=torch.zeros_like(timestep).to(clean_images.device)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler1.add_noise(clean_images, noise, timesteps)
            noisy_images2 = noise_scheduler2.add_noise(clean_images, noise2, timesteps2)
            with accelerator.accumulate(model):
                # Predict the noise residual
                z_pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]
                model2_input=torch.cat([noisy_images2,z_pred],dim=1)
                noise_pred2 = model2(model2_input, timesteps2, class_labels, return_dict=False)[0]
                # x0_pred = noise_scheduler.step(noise_pred2, timesteps2, noisy_images2)
                # loss = F.mse_loss(x0_pred, clean_images)
                loss = F.mse_loss(noise_pred2, noise2)
                # loss=loss+loss2
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        # TODO add id2labels
        if accelerator.is_main_process:
            transforms1=accelerator.unwrap_model(model)
            transforms2=accelerator.unwrap_model(model2)
            vae=accelerator.unwrap_model(vae)
            pipeline = DiffDiffPipeline(transformer1=transforms1, transformer2=transforms2, vae=vae, scheduler1=noise_scheduler1,scheduler2=noise_scheduler2)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                upload_thread = threading.Thread(target=save_model, args=(config, repo_id, epoch, pipeline))
                upload_thread.start()
                upload_thread.join()  # 等待上传完成

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    patch_size = 2
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 600000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 300
    save_model_epochs = 1000
    mixed_precision = "bf16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/0810_diffdiff-butterflies-64"  # the model name locally and on the HF Hub
    
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "wangyanhui666/test_diffdiff3"  # the name of the repository to create on the HF Hub
    hub_private_repo = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["image"]):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# fig.savefig("output_image.png")  # 保存到当前目录，文件名为 output_image.png
# plt.close(fig)  # 关闭图像以释放内
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model=DiT_XL_2(sample_size=config.image_size,in_channels=3,out_channels=3)
model2=DiT_S_2(sample_size=config.image_size,in_channels=6,out_channels=3)
vae=Vae()
# check the sample image shape matches the model output shape
sample_image = dataset[0]["images"].unsqueeze(0)
timestep=torch.LongTensor([0])
classlabel=torch.LongTensor([0])
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=timestep,class_labels=classlabel).sample.shape)

# create a scheduler
noise_scheduler1 = DDPMScheduler(num_train_timesteps=1000)
noise_scheduler2 = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler1.add_noise(sample_image, noise, timesteps)
image=Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
image.save("noisy_image.png")

# traing the model
optimizer = torch.optim.AdamW([{'params': model.parameters()},
                {'params': model2.parameters()}], lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


args = (config, model, model2, vae, noise_scheduler1,noise_scheduler2, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=4)
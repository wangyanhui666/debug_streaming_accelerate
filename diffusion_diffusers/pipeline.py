from typing import Dict, List, Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline, AutoencoderKL, DiTTransformer2DModel, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

class DiffDiffPipeline(DiffusionPipeline):
    r"""
    Pipeline for diffusion over diffusion.

    Parameters:
        transformer1 ([`DiTTransformer2DModel`]):
            A large class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.
        transformer2 ([`DiTTransformer2DModel`]):
            A small class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "transformer1->transformer2->vae"

    def __init__(self,
                    transformer1: DiTTransformer2DModel,
                    transformer2: DiTTransformer2DModel,
                    vae: AutoencoderKL,
                    scheduler1: KarrasDiffusionSchedulers,
                    scheduler2: KarrasDiffusionSchedulers,
                    id2label: Optional[Dict[int, str]] = None):
            super().__init__()
            self.register_modules(transformer1=transformer1, transformer2=transformer2, vae=vae, scheduler1=scheduler1, scheduler2=scheduler2)
    
            # create a imagenet -> id dictionary for easier use
            self.labels = {}
            if id2label is not None:
                for key, value in id2label.items():
                    for label in value.split(","):
                        self.labels[label.lstrip().rstrip()] = int(key)
                self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]
    
    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale1: float = 4.0,
        guidance_scale2: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps1: int = 50,
        num_inference_steps2: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.
        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps1 (`int`, *optional*, defaults to 250):
                The number of denoising steps of transformer1. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            num_inference_steps2 (`int`, *optional*, defaults to 250):
                The number of denoising steps of transformer2. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        latent_size=self.transformer1.config.sample_size
        latent_channels = self.transformer1.config.in_channels

        latents1=randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer1.dtype,
        )

        latent_model_input1 = torch.cat([latents1] * 2) if guidance_scale1 > 1 else latents1

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input1 = torch.cat([class_labels, class_null], 0) if guidance_scale1 > 1 else class_labels
        class_labels_input2 = torch.cat([class_labels, class_null], 0) if guidance_scale2 > 1 else class_labels

        # set step values

        self.scheduler1.set_timesteps(num_inference_steps1)
        self.scheduler2.set_timesteps(num_inference_steps2)
        for t1 in self.progress_bar(self.scheduler1.timesteps):
            if guidance_scale1 > 1:
                half = latent_model_input1[: len(latent_model_input1) // 2]
                latent_model_input1 = torch.cat([half, half], dim=0)
            latent_model_input1 = self.scheduler1.scale_model_input(latent_model_input1, t1)

            timesteps1 = t1
            if not torch.is_tensor(timesteps1):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps=latent_model_input1.device.type=="mps"
                if isinstance(timesteps1, float):
                    dtype=torch.float32 if is_mps else torch.float64
                else:
                    dtype=torch.int32 if is_mps else torch.int64
                timesteps1=torch.tensor([timesteps1], dtype=dtype, device=latent_model_input1.device)
            elif len(timesteps1.shape)==0:
                timesteps1=timesteps1[None].to(latent_model_input1.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps1 = timesteps1.expand(latent_model_input1.shape[0])
            # predict Intermediate variable model_output1
            z_pred = self.transformer1(
                latent_model_input1, timestep=timesteps1, class_labels=class_labels_input1
            ).sample

            # start model2 diffusion process
            latents2=randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer2.dtype,
            )
            latent_model_input2 = torch.cat([latents2] * 2) if guidance_scale2 > 1 else latents2
            for t2 in self.progress_bar(self.scheduler2.timesteps):
                if guidance_scale2 > 1:
                    half = latent_model_input2[: len(latent_model_input2) // 2]
                    latent_model_input2 = torch.cat([half, half], dim=0)
                latent_model_input2 = self.scheduler2.scale_model_input(latent_model_input2, t2)

                timesteps2 = t2
                if not torch.is_tensor(timesteps2):
                    is_mps=latent_model_input2.device.type=="mps"
                    if isinstance(timesteps2, float):
                        dtype=torch.float32 if is_mps else torch.float64
                    else:
                        dtype=torch.int32 if is_mps else torch.int64
                    timesteps2=torch.tensor([timesteps2], dtype=dtype, device=latent_model_input2.device)
                elif len(timesteps2.shape)==0:
                    timesteps2=timesteps2[None].to(latent_model_input2.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps2 = timesteps2.expand(latent_model_input2.shape[0])
                # predict noise model_output2
                concat_model_input2=torch.cat([latent_model_input2,z_pred],dim=1)
                noise_pred2 = self.transformer2(
                    concat_model_input2, timestep=timesteps2, class_labels=class_labels_input2
                ).sample
                # perform guidance
                if guidance_scale2 > 1:
                    eps, rest = noise_pred2[:, :latent_channels], noise_pred2[:, latent_channels:]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                    half_eps = uncond_eps + guidance_scale2 * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)

                    noise_pred2 = torch.cat([eps, rest], dim=1)
                # learned sigma
                if self.transformer2.config.out_channels // 2 == latent_channels:
                    model_output2, _ = torch.split(noise_pred2, latent_channels, dim=1)
                else:
                    model_output2=noise_pred2

                # compute previous image: x_t -> x_t-1
                latent_model_input2=self.scheduler2.step(model_output2,t2,latent_model_input2).prev_sample

            if guidance_scale2 > 1:
                latents2, _ = latent_model_input2.chunk(2, dim=0)
            else:
                latents2 = latent_model_input2
            # end of scheduler2 get x0_pred latents2
            # get noise pred of full model from xt and x0_pred
            # TODO this maybe only works for DDPM DDIM
            alpha_prod_t1 = self.scheduler1.alphas_cumprod[t1]
            beta_prod_t1 = 1 - alpha_prod_t1
            noise_pred1= (latent_model_input1 - alpha_prod_t1**(0.5) * latents2) / beta_prod_t1**(0.5)

            # perform guidance
            if guidance_scale1 > 1:
                eps, rest = noise_pred1[:, :latent_channels], noise_pred1[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale1 * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred1 = torch.cat([eps, rest], dim=1)
            
            # learned sigma
            if self.transformer1.config.out_channels // 2 == latent_channels:
                model_output1, _ = torch.split(noise_pred1, latent_channels, dim=1)
            else:
                model_output1=noise_pred1

            # compute previous image: x_t -> x_t-1
            latent_model_input1=self.scheduler1.step(model_output1,t1,latent_model_input1).prev_sample
        
        if guidance_scale1 > 1:
            latents1, _ = latent_model_input1.chunk(2, dim=0)
        else:
            latents1 = latent_model_input1
        # end of scheduler1 get x_0_pred latents1

        # decode the latents to images
        latents1 = 1 / self.vae.config.scaling_factor * latents1
        samples = self.vae.decode(latents1).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
            
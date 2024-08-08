from diffusers import DiTTransformer2DModel,AutoencoderKL,ModelMixin,ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from torch import nn
import os
import json
def DiT_XL_2(**kwargs):
    return DiTTransformer2DModel(num_layers=28, attention_head_dim=72,patch_size=2,num_attention_heads=16,**kwargs)

def DiT_L_2(**kwargs):
    return DiTTransformer2DModel(num_layers=24, attention_head_dim=64,patch_size=2,num_attention_heads=16,**kwargs)

def DiT_B_2(**kwargs):
    return DiTTransformer2DModel(num_layers=12, attention_head_dim=64,patch_size=2,num_attention_heads=12,**kwargs)

def DiT_S_2(**kwargs):
    return DiTTransformer2DModel(num_layers=12, attention_head_dim=64,patch_size=2,num_attention_heads=6,**kwargs)



import torch
import torch.nn as nn


class VAE(ModelMixin, ConfigMixin,FromOriginalModelMixin):
    @register_to_config
    def __init__(self, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def decode(self, latents):
        # This is a mock decode function for demonstration purposes
        class Sample:
            def __init__(self, data):
                self.data = data

            @property
            def sample(self):
                return self.data
        
        # Assuming the decode function reverses the scaling operation
        return Sample(latents * self.config.scaling_factor)

def Vae():
    vae = VAE()
    return vae

DiT_models = {
    'DiT-XL/2': DiT_XL_2,
}
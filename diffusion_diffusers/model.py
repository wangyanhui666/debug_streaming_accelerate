from diffusers import DiTTransformer2DModel
from torch import nn
def DiT_XL_2(**kwargs):
    return DiTTransformer2DModel(num_layers=28, attention_head_dim=72,patch_size=2,num_attention_heads=16,**kwargs)



class VAEConfig(nn.Module):
    def __init__(self, scaling_factor):
        super(VAEConfig, self).__init__()
        self.scaling_factor = scaling_factor

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config

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

def Vae(**kwargs):
    scaling_factor = 1.0
    vae_config = VAEConfig(scaling_factor)
    vae = VAE(vae_config)
    return vae

DiT_models = {
    'DiT-XL/2': DiT_XL_2,
}
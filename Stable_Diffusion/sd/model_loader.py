# file to load weights from checkpoints
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAEDecoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAEDecoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)

    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP(49408, 768, 77,12).to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
# the pipeline which integrates all the modules

import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# width and height of images, latent feature 
WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8
'''
prompt : the text prompt fed to diffusion model
uncond_prompt : negative prompt - the thing which is not be present in the image
input_image : the input image which may be used for inpainting [modifying the given image based on prompts]
cfg : classifier free guidance
n_inference_steps : number of steps in the diffusion process
strength : a measure of how strongly the input image should be modified.ie how much creative can the model get based on the noise added to the encoder's latent space
'''
def generate(prompt : str, uncond_prompt : str, input_image : None,
             do_cfg : bool = True, cfg_scale =0.75, strength : float = 0.8, 
             sampler_name = "ddpm", n_inference_step = 50, models = {},seed=None,
              device = None, idle_device = None, tokenizer=None):
    with torch.no_grad():
        if not (0 <= strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x : x.to(idle_device)
        else:
            to_idle = lambda x : x
        
        generator = torch.Generator(device = device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        
        clip = models["clip"]
        clip.to(device)
        # if classifier free guidance is required!
        if do_cfg:
            # tokenization of conditional prompt with max_length 77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # batch_size, seq_len
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # batch_size, seq_len, context_dim
            cond_context = clip(cond_tokens)

            # convert negative prompt to tokens
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # batch_size, seq_len
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # batch_size, seq_len, context_dim
            uncond_context = clip(uncond_tokens)
            # concatenate conditional and unconditional prompts
            # 2*batch_size, seq_len, context_dim
            context = torch.cat([cond_context, uncond_context])
        
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # batch_size, seq_len
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # batch_size, seq_len, context_dim
            context = clip(tokens)
        
        # push the clip model to idle device
        to_idle(clip)

        # use the ddpm sampler to set the inference steps
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_step)

        else:
            raise ValueError("Unknown sampler : {sampeler_name}")
        # latent feature map size   
        latent_shape = 1, 4, LATENT_HEIGHT, LATENT_WIDTH

        # if input image is given
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            #resize the input image to HEIGTH, WIDTH set in the encoder
            input_image = input_image.resize((WIDTH, HEIGHT))
            input_image = np.array(input_image)
            # image height, image width, channels
            input_image_tensor = torch.tensor(input_image, dtype=torch.float32, device=device)
            # rescale the input image to the range [-1, 1] suitable for encoder
            input_image_tensor = scale(input_image_tensor, (0,255),(-1,1))
            # height, width, channels -> batch_size, height, width
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # batch_size, height, width, channels - > batch_size, channels, height, width
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            # generate the noise for the encoder to sample a latent feature map 
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # batch_size, 4, height, width
            latents = encoder(input_image_tensor, encoder_noise)

            # add noise to the latent feature map for the diffusion model to get creative in sampling from the distribution

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        
        else:
            # no input image, hence we pass the noise alone as output of the encoder to the diffusion model
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        # loop throught the diffusion timesteps
        for t, timestep in enumerate(timesteps):
            # (1, 320) -> shape required by the time embedding model
            time_embedding = get_time_embedding(timestep).to(device)
            # pass the latent feature map from the encoder to the diffusion model
            model_input = latents

            if do_cfg:
                # replicate the model twice one for conditional prompt, one for unconditional prompt
                model_input = model_input.repeat(2,1,1,1)
            # pass through the unet which combines input image's latent space, prompt embeddign and timestep embedding
            # batch_size, 4, height, width -> batch_size, 4, height, width
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            # reverse process of the diffusion model
            # batch_size, 4, latent_height, latent_width -> batch_size, 4, latent_height, latent_width
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        # batch_size, 4, latent_height, latent_width -> batch_size,3,height,width
        images = decoder(latents)
        to_idle(decoder)
        #batch_size,3,height,width
        images = rescale(images , (-1,-1),(0,255), clamp=True)
        #batch_size,3,height,width -> batch_size,height,width,3
        images = images.permute(0,2,3,1)
        images = images.to("cpu", dtype=torch.uint8).numpy()
        return images[0]
    
# function to rescale the images

def rescale(input, old_range, new_range,clamp=False):
    # get the old and new mins and maxs
    old_min, old_max = old_range
    new_min, new_max = new_range
    # rescale the image to have new range
    x-=old_min
    x*= ((new_max-new_min)/(old_max-old_min))
    x += new_min
    if clamp:
        # clamp within the new range
        x = torch.clamp(x, new_min, new_max)
    
    return x
# function similar to the transformer's positional time encoding
def get_time_embedding(timestep):
    # shape : (160,)
    freqs = torch.pow(10000, -torch.arange(start=0,end=160,dtype=torch.float32)/160)
    #shape : (1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:,None] * freqs[None] # increase the rank by one

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


    
















        

        




        


    


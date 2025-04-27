import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD = [0.5,0.5,0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_length, image_token):
     # bos token in pali gemma is newline character \n. it is explicitly stated 
     # to prevent the \n being merged with other tokens.
     return f"{image_token * image_seq_length}{bos_token}{prefix_prompt}\n"

def resize(image, size, resample, reducing_gap=None):
     h,w = size
     resized_image = image.resize((w,h), resample=resample,reducing_gap=reducing_gap)
     return resized_image

def rescale(image, scale, dtype=np.float32):
     rescaled_image = image * scale
     return rescaled_image.astype(dtype)

def normalize(image, mean, std):
    mean = np.array(mean, dtype=image.dtype)
    std= np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image
     

def preprocess_images(images, size, resample, rescale_factor, image_mean, image_std):
        height, width = size
        images = [resize(image, size=(height, width),resample=resample) for image in images]
        images = [np.array(image) for image in images]
        images = [rescale(image, scale=rescale_factor) for image in images]
        images = [normalize(image, mean = image_mean, std = image_std) for image in images]
        # h,w,c -> c,h,w
        images = [image.transpose(2,0,1) for image in images]
        return images

class PaliGemmaProcessor(nn.Module):
    IMAGE_TOKEN = "<image>"
    def __init__(self, tokenizer, num_image_tokens,image_size):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens":[self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)] # used for object detection
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)] # used for semantic segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # Beginning and end of sequence tokens
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
    
    
    def __call__(self, images, text, padding="longest", truncation=True):
        # currently we care only about one image and one text. if we want this to work for 
        # multiple images and texts, we need to change the code.
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} for {len(text)}"

        pixel_values = preprocess_images(images,
                                              size=(self.image_size, self.image_size),
                                              resample = Image.Resampling.BICUBIC,
                                              rescale_factor = 1./255,
                                              image_mean = IMAGENET_STANDARD_MEAN,
                                                image_std = IMAGENET_STANDARD_STD,                                       
        )
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        # prepend a self.image_seq_length tokens to the prompt
        input_strings = [
             add_image_tokens_to_prompt(
                  prefix_prompt = prompt,
                  bos_token = self.tokenizer.bos_token,
                  image_seq_length = self.image_seq_length,
                  image_token = self.IMAGE_TOKEN
             ) for prompt in text
        ]

        # return the input ids and attention mask
        inputs = self.tokenizer(
             input_strings,
             return_tensors = "pt",
             padding=padding,
             truncation=truncation,
        )

        return_data = {"pixel_values":pixel_values,**inputs}
        return return_data
             










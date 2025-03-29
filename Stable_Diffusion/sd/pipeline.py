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


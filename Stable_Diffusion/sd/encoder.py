# this file contains the implementation of Encoder block of the 
# Variational AutoEncoder block

import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAEAttentionBlock, VAEResidualBlock

# VAE encoder block
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # constructor of nn.Sequential class
        super().__init__(
            # Declaration of a bunch of conv layers which constitute the encoder block
            # (batch_size, 3, H, W) -> (batch_size, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding = 1), # padding to preserve shape of image

            VAEResidualBlock(128, 128), 
            VAEResidualBlock(128, 128), 
            # (batch_size, 128, H, W) -> (batch_size, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # stride=2 reduces the shape of image
            # number of channels doubled
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),
            # (batch_size, 128, H/2, W/2) -> (batch_size, 128, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # number of channels doubled
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),
            # (batch_size, 128, H/4, W/4) -> (batch_size, 128, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            # VAE attention block -> similar to the attention in language models
            # individual pixels in this latent image is taken as tokens.

            VAEAttentionBlock(512),

            VAEResidualBlock(512, 512),

            # group normalization applied on the tensor
            #(batch_size, 512, H/8, W/8) -> (batch_size, 512, H/8, W/8)
            nn.GroupNorm(32,512),

            # Swiglu activation function -> works better than relu in practice
            nn.SiLU(),

            # reduction in num_channels to 8, preserving shape
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8,8,kernel_size=1, padding=0),
        )
        self.variance_constant = 0.18215
    
    def forward(self, x, noise):
        # x : (batch_size, Channel, height, width)
        # noise : (batch_size, 4, height/8, width/8)

        for module in self:
            # pick conv layer which has stride = 2
            if getattr(self, "stride", None) == (2,2):
                # Pad -> (pad_left, pad_right, pad_top, pad_bottom)
                # here pad_right and pad_bottom is enabled
                # outshape in a conv layer = (w - k + 2p)/s + 1, if padding is enabled both sides
                # if it is enabled one side both sides and top-bottom,
                # outshape = (w-k+p)/s + 1, which prevents rounding in the calculation of (w-k+p)/s for kernel size of 3
                 
                x = F.pad(x, (0,1,0,1))
            x = module(x)
        
        # getting the mean and log_variance from the downsampled feature map
        # (b,8,h/8,w/8) -> two tensors of shape (b,4,h/8,w/8)
        mean, log_variance = torch.chunk(x, 2,dim=1)
        # convert log_variance to variance after clamping
        log_variance = torch.clamp(log_variance, -30, 20)

        variance = torch.exp(log_variance)

        # computation of standard deviation
        std = torch.sqrt(variance)

        # Transform N(0,1) to N(mean, std)
        # (batch_size, 4, h/8, w/8) -> (batch_size, 4, h/8, w/8)
        x = mean + std * noise

        # scale by a unitvariance constant 
        # for this particular architecture, the image goes through layers of conv
        # so , there is a change in variance of the feature map from the original image variance
        # it is good to have the variance scaled back for better image generation in the original image space

        x *= self.variance_constant

        return x








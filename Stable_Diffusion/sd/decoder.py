# this file contains the implementation of Decoder block of the 
# Variational AutoEncoder block

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import selfAttention

# VAE's attention block 
class VAEAttentionBlock(nn.Module):
    def __init__(self, channels : int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = selfAttention(channels, 1) # d_embed = num_channels, num_heads = 1
    
    def forward(self, x):
        # x : (batch_size, features, height, width)
        # residual skip connection
        residue = x

        x = self.groupnorm(x)

        n,c,h,w = x.shape

        x = x.view((n,c,h*w))
        # shape changed to (batch_size, h*w, num_channels)
        x = x.transpose(-1,-2)
        # apply attention to the feature map
        x = self.attention(x)
        # transpose back to original shape -> (batch_size, num_channels, h*w)
        x = x.transpose(-1,-2)

        x = x.view((n,c,h,w))

        # add the input before the attention block -> skip connection
        x = residue + x

        return x

# residual block which is used in encoder and decoder 
class VAEResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x
        # b,c_in,h,w -> b,c_in,h,w
        x = self.groupnorm1(x)
        # b,c_in,h,w -> b,c_in,h,w
        x = F.silu(x)
        # b,c_in,h,w -> b,c_out,h,w
        x = self.conv1(x)
        # b,c_out,h,w -> b,c_out,h,w
        x = self.groupnorm2(x)
        # b,c_out,h,w -> b,c_out,h,w
        x = F.silu(x)
        # b,c_out,h,w -> b,c_out,h,w
        x = self.conv2(x)

        return x + self.residual(residue)

# decoder block of the VAE
class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
        # these blocks mainly reverse the effect caused in the encoder to reconstruct back the original image
        # (batch_size, 4, h/8,w/8) -> (batch_size, 4, h/8,w/8)
        nn.Conv2d(4,4,kernel_size=1, padding=0),
        # (batch_size, 4, h/8,w/8) -> (batch_size, 512, h/8,w/8)
        nn.Conv2d(4,512,kernel_size=3, padding=1),
        # (batch_size, 512, h/8,w/8)
        VAEResidualBlock(512, 512), 
        # (batch_size, 512, h/8,w/8)
        VAEAttentionBlock(512),
        # (batch_size, 512, h/8, w/8)
        VAEResidualBlock(512, 512), 
        VAEResidualBlock(512, 512), 
        VAEResidualBlock(512, 512), 
        VAEResidualBlock(512, 512), 

        # to upscale the image, simple nn.Upsample api coupled with conv2d is used instead of deconv as used in encoder
        # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/4, w/4)
        nn.Upsample(scale_factor=2),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # (batch_size, 512, h/8, w/8)
        VAEResidualBlock(512, 512),
        VAEResidualBlock(512, 512),
        VAEResidualBlock(512, 512),
        # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/2, w/2)
        nn.Upsample(scale_factor=2),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),

        # (batch_size, 512, h/2, w/2) -> (batch_size, 256, h/2, w/2)
        VAEResidualBlock(512, 256),
        VAEResidualBlock(256, 256),
        VAEResidualBlock(256, 256),
        # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h, w)
        nn.Upsample(scale_factor=2),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        # (batch_size, 256, h, w) -> (batch_size, 128, h, w)
        VAEResidualBlock(256, 128),
        VAEResidualBlock(128, 128),
        VAEResidualBlock(128, 128),

        # group normalization
        nn.GroupNorm(32, 128),
        # swiglu activation function
        nn.SiLU(),
        # batch_size, 128, h, w -> batch_size, 3, h, w [image resolution]
        nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
        self.variance_constant = 0.18215
    
    def forward(self, x):
        # x - latent feature map : (batch_size, 4, height/8, width/8)
        # reverse scaling to revert the effect caused in encoder
        x /= self.variance_constant
        for module in self:
            x = module(x)
        # input image resolution -> (batch_size, 3, height, width)
        return x












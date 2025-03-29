# the U-net model which performs the diffusion of the image

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import selfAttention, crossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim : int):
        super().__init__()
        self.linear1 = nn.Linear(time_dim, 4*time_dim)
        self.linear2 = nn.Linear(4*time_dim, 4*time_dim)
    
    def forward(self, x):
        # 1, time_dim -> 1, 4*time_dim
        x = self.linear1(x)
        # swiglu activation function
        x = F.silu(x)
        # 1, 4*time_dim -> 1, 4*time_dim
        x = self.linear2(x)
        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, n_time : 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels,kernel_size=1, padding=0)
        
    def forward(self, feature, time):
        # feature : batch_size, in_channels, height, width
        # time : 1, 1280
        residue = feature
        # batch_size, in_channels, height, width -> batch_size, in_channels, height, width
        feature = self.groupnorm_feature(feature)
        # batch_size, in_channels, height, width -> batch_size, in_channels, height, width
        feature = F.silu(feature)
        # batch_size, in_channels, height, width -> batch_size, out_channels, height, width
        feature = self.conv_feature(feature)
        # 1, 1280 -> 1,1280
        time = F.silu(time)
        # 1,1280 -> 1,out_channels
        time = self.linear_time(time)

        # broadcast time to the dimension of feature and then element wise add
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        # batch_size, out_channels, height, width -> batch_size, out_channels, height, width
        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual(residue)
    

class UNET_AttentionBlock(nn.Module):
    def __init__(self,n_heads : int, embed_dim : int, d_context : int = 768):
        super().__init__()
        self.d_context = d_context
        channels = n_heads * embed_dim

        self.groupnorm = nn.GroupNorm(32, channels)
        self.convinput = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = selfAttention(channels, n_heads, in_proj_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = crossAttention(channels, n_heads, d_context, in_proj_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)

        self.linear_geglu1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu2 = nn.Linear(4 * channels, channels)

        self.convoutput = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x : batch_size, channels, height, width
        # context : batch_size, seq_len, d_context
        # residual connection for the entire block
        residue_long = x
        # batch_size, channels, height, width -> batch_size, channels, height, width
        x = self.groupnorm(x)

        x = self.convinput(x)

        n, c, h, w = x.shape
        # batch_size, channels, height, width ->  batch_size, channels, height*width 
        x = x.view((n, c, h*w))
        # batch_size, channels, height*width -> batch_size, height*width, channels
        x = x.transpose(-1, -2)
        # residual connection for the smaller sub-block
        residue_short = x
        # normalization + self attention
        x = self.layernorm1(x)
        # batch_size, height*width, channels -> batch_size, height*width, channels
        x = self.attention1(x)
        # adding the skip connection
        x = x + residue_short

        residue_short = x
        # normalization + cross attention between features and context from CLIP
        x = self.layernorm2(x)
        #batch_size, height*width, channels -> batch_size, height*width, channels
        x = self.attention2(x, context)

        x = x + residue_short

        residue_short = x

        x = self.layernorm3(x)

        # batch_size, height*width, channels -> two tensors of shape batch_size,  height*width, 4*channels
        x, gate = self.linear_geglu1(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)
        #batch_size,  height*width, 4*channels -> batch_size, height * width , channels
        x = self.linear_geglu2(x)

        x += residue_short

        # convert the tensors back to original shape as input
        # batch_size, height * width , channels -> batch_size, channels, height * width
        x = x.transpose(-1, -2)
        # batch_size, channels, height * width -> batch_size, channels, height, width
        x = x.view((n,c,h,w))

        return self.convoutput(x) + residue_long       

# since we inherit from nn.Sequantial, we do not need to write layers 
# explicitly , we can directly construct the layers and write forward
class SwitchSequential(nn.Sequential):
   
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)

class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
   
    def forward(self,x):
        # x : batch_size, num_channels, height, width
        # batch_size, num_channels, height, width -> batch_size, num_channels, height*2, width*2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
           
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # batch_size, 4, height/8, width/8 -> batch_size, 320, height/8, width/8
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            # batch_size, 320, height/8, width/8 -> batch_size, 320, height/8, width/8 -> batch_size, 320, height/8, width/8
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),
            # batch_size, 320, height/8, width/8 -> # batch_size, 320, height/16, width/16
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # batch_size, 320, height/16, width/16 -> batch_size, 640, height/16, width/16  
            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),
            # batch_size, 640, height/16, width/16 -> # batch_size, 640, height/32, width/32
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            #batch_size, 640, height/32, width/32 -> # batch_size, 1280, height/32, width/32
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8,160)),

            #batch_size, 1280, height/32, width/32 -> # batch_size, 1280, height/32, width/32
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8,160)),
            #batch_size, 1280, height/32, width/32 -> # batch_size, 1280, height/64, width/64
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            # batch_size, 1280, height/64, width/64
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # batch_size, 1280, height/64, width/64
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ]),
        
        self.bottle_neck = SwitchSequential(
            # batch_size, 1280, height/64, width/64 -> batch_size, 1280, height/64, width/64
            UNET_ResidualBlock(1280, 1280),
            # batch_size, 1280, height/64, width/64 -> batch_size, 1280, height/64, width/64
            UNET_AttentionBlock(8,160),
            # batch_size, 1280, height/64, width/64 -> batch_size, 1280, height/64, width/64
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # batch_size, 2560, height/64, width/64 -> batch_size, 1280, height/64, width/64
            # the reason for having 2560 here is during skip connection adding, the encoders outputs are concatenated in the decoder layers
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            # batch_size, 2560, height/64, width/64 -> batch_size, 1280, height/64, width/64           
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            # batch_size, 2560, height/64, width/64 -> batch_size, 1280, height/64, width/64 -> batch_size, 1280, height/32, width/32
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            # batch_size, 2560, height/32, width/32 -> # batch_size, 1280, height/32, width/32
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            # batch_size, 2560, height/32, width/32 -> # batch_size, 1280, height/16, width/16
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8,160), Upsample(1280)),
            # batch_size, 1920, height/16, width/16 -> batch_size, 640, height/16, width/16
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            # batch_size, 1280, height/16, width/16 -> batch_size, 640, height/16, width/16
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8,80)), 
            # batch_size, 960, height/16, width/16 -> batch_size, 640, height/8, width/8
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8,80), Upsample(640)),
            # batch_size, 960, height/8, width/8 -> batch_size, 320, height/8, width/8
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8,40)),
            # batch_size, 640, height/8, width/8 -> batch_size, 320, height/8, width/8
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            # batch_size, 640, height/8, width/8 -> batch_size, 320, height/8, width/8
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])
        
    def forward(self, x, context, time):
        # latent feature map : (batch_size, 4, height/8, width/8)
        # context : the embedding from CLIP : (batch_size, seq_len, embed_dim)
        # time : (1,1280)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
        
        x = self.bottle_neck(x, context, time)

        for layers in self.decoders:
            # concat with the encoder from the bottom
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)
        
        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x : batch_size, 320, height/8, width/8
        x = self.group_norm(x)
        #batch_size, 320, height/8, width/8 -> batch_size, 320, height/8, width/8
        x = F.silu(x)
        # batch_size, 320, height/8, width/8 -> batch_size, 4, height/8, width/8
        x = self.conv(x)

        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)


    def forward(self, latent, context, time):
        # latent feature map : (batch_size, 4, height/8, width/8)
        # context : the embedding from CLIP : (batch_size, seq_len, embed_dim)
        # time : embedding from time (1,320)

        # 1, 320 -> 1,1280
        time = self.time_embedding(time)
        # batch_size, 4, height/8, width/8 -> batch_size, 320, height/8, width/8
        output = self.unet(latent, context, time)
        # batch_size, 320, height/8, width/8 -> batch_size, 4, height/8, width/8
        output = self.final(output)

        return output




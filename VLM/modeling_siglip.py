import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size = 768,
            intermediate_size = 3072,
            num_hidden_layers = 12,
            num_attention_heads = 12,
            num_image_channels = 3,
            image_size = 224, # square image
            patch_size = 16,
            layer_norm_eps = 1e-6,
            attention_dropout = 0.0,
            num_image_tokens : int = None,
            **kwargs

    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_image_channels = num_image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SigLipVisionEmbedding(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels = config.num_image_channels,
            out_channels = self.embed_dim, 
            kernel_size = config.patch_size,
            stride = config.patch_size,
            padding = "valid"            
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_pos = self.num_patches # for positional embedding
        self.position_embeddings = nn.Embedding(self.num_pos, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_pos).expand((1,-1)),
                             persistent = False)

    
    def forward(self, pixel_values):
        # batch_size, num_channels, height, width -> batch_size, num_patches, embed_dim
        _, _, height, width = pixel_values.shape
        # b,c,h,w -> b,embed_dim,h//p, w//p
        patch_embed = self.patch_embeddings(pixel_values)
        patch_embed = patch_embed.flatten(2).transpose(1,2) # b,embed_dim,h,w -> b,h*w,embed_dim

        patch_embed = patch_embed + self.position_embeddings(self.position_ids)

        #[batch_size, num_patches, embed_dim]
        return patch_embed
    
class SiglipAttention(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_state):
        # b, num_patches, embed_dim
        b, seq_len, embed_dim = hidden_state.shape
        query_state = self.q_proj(hidden_state) # b, num_patches, embed_dim
        key_state = self.k_proj(hidden_state)
        value_state = self.v_proj(hidden_state)

        query_state = query_state.view(b,seq_len, self.num_heads, self.head_dim).transpose(1,2) # b, num_heads, seq_len, head_dim
        key_state = key_state.view(b,seq_len, self.num_heads, self.head_dim).transpose(1,2) # b, num_heads, seq_len, head_dim
        value_state = value_state.view(b,seq_len, self.num_heads, self.head_dim).transpose(1,2) # b, num_heads, seq_len, head_dim

        attn_weights = torch.matmul(query_state, key_state.transpose(2,3))*self.scale # b,num_heads, seq_len, seq_len
        attn_weights = torch.softmax(attn_weights, dim = -1,dtype=torch.float32).to(query_state.dtype) # b,num_heads, seq_len, seq_len
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # b, num_heads, seq_len, seq_len @ b, num_heads, seq_len, head_dim -> b, num_heads, seq_len, head_dim
        attn_output = torch.matmul(attn_weights, value_state)

        attn_output = attn_output.transpose(1,2).contiguous()
        # b, num_heads, emded_dim
        attn_output = attn_output.reshape(b, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_state):
        # b, num_patches, embed_dim -> b, num_patches, intermediate_size
        hidden_state = self.fc1(hidden_state)
        # GELU activation function
        hidden_state = F.gelu(hidden_state, approximate="tanh")
        # b, num_patches, intermediate_size -> b, num_patches, embed_dim
        hidden_state = self.fc2(hidden_state)
        return hidden_state


# based on the transformer encoder in Attention is all you need
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layernorm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.MLP = SiglipMLP(config)
        self.layernorm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
    
    
    def forward(self, hidden_state):
        # b, num_patches, embed_dim
        residual = hidden_state
        # layer norm1
        hidden_state = self.layernorm1(hidden_state)
        # self attention : b, num_patches, embed_dim -> b, num_patches, embed_dim
        hidden_state,_ = self.self_attn(hidden_state)
        # residual connection
        hidden_state = residual + hidden_state

        residual = hidden_state
        # mlp layer : b, num_patches, embed_dim -> b, num_patches, embed_dim
        hidden_state = self.MLP(hidden_state)
        # layer norm2 
        hidden_state = self.layernorm2(hidden_state)
        # skip connection
        hidden_state = residual + hidden_state
        return hidden_state

class SiglipEncoder(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_state):
        # b, num_patches, embed_dim
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
    
    def forward(self, pixel_values):
        hidden_state = self.embeddings(pixel_values)
        
        encoded_state = self.encoder(hidden_state)

        out = self.post_layernorm(encoded_state)

        return out





class SiglipVisionModel(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values):
        # batch_size, num_channels, height, width -> batch_size, num_patches, embed_dim
        return self.vision_model(pixel_values)
    







        
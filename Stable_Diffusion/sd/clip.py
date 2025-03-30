# building the clip embedding block

# import the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import selfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size : int, embed_dim : int, n_tokens : int):
        super().__init__()
        # embedding layer for the tokens with vocab_size length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # a learnable tensor which encodes the positional information of tokens
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens, embed_dim)))
    
    def forward(self, x):
        # batch_size, seq_len -> batch_size, seq_len, embed_dim
        x = self.token_embedding(x)
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, embed_dim
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_heads : int, embed_dim : int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = selfAttention(embed_dim, n_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4*embed_dim, embed_dim)
    
    def forward(self, x):
        # residual connection
        # batch_size, seq_len, embed_dim
        residue = x
        # batch_size, seq_len, embed_dim
        x = self.layer_norm1(x)
        # batch_size, seq_len, embed_dim
        x = self.attention(x, causal_mask=True)

        x += residue

        residue = x
        # batch_size, seq_len, embed_dim
        x = self.layer_norm2(x) 
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, 4*embed_dim
        x = self.linear1(x)
       #batch_size, seq_len, 4*embed_dim
       # activation function applied
       # GELU stands for Guassian Error Linear unit and is given by:
       # GELU(x) = x * phi(x) where phi(x) = 0.5(1+erf(x/sqrt(2))) -> cumulative gaussian distribution function
       # computation of erf is expensive and hence it is approximated by the function, sigmoid(1.702*x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU function
        # batch_size, seq_len, 4*embed_dim -> batch_size, seq_len, embed_dim
        x = self.linear2(x)

        x += residue
        
        return x

class CLIP(nn.Module):
    def __init__(self, vocab_size : int, embed_dim : int, n_tokens : int, n_layers : int):
        super().__init__()
        self.embedding = CLIPEmbedding(vocab_size, embed_dim, n_tokens)
        self.layers = nn.ModuleList([
            CLIPLayer(12, embed_dim) for i in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x : torch.LongTensor) -> torch.FloatTensor:
        x = x.type(torch.long)
        # batch_size, seq_len -> batch_size, seq_len, embed_dim
        state = self.embedding(x)
        # apply the clip layer in succession : similar to transformer's encoder layer
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, embed_dim
        for layer in self.layers:
            state = layer(state)
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, embed_dim
        state = self.layer_norm(state)
        
        return state
        








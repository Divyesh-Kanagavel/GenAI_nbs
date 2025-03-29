# implementation of self attention and cross attention blocks to be used 
# in the encoder and the decoder of the VAE

#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# self attention block
class selfAttention(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, in_proj_bias : bool = True,
                out_proj_bias : bool = True):
        super().__init__()
        # Wq, Wk, Wv combined into one matrix in the output of the layer
        self.in_proj = nn.Linear(
            embed_dim, embed_dim * 3, bias = in_proj_bias
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias = out_proj_bias
        )
        self.n_heads = num_heads
        self.d_head = embed_dim // num_heads
    
    def forward(self, x : torch.Tensor, causal_mask : bool = False) -> torch.Tensor:
        # extract the input shape from input tensor
        input_shape = x.shape

        # split input shape into batch_size, seq_len, embed_dim
        batch_size, seq_len, embed_dim = input_shape

        # split the shape into multi heads
        inter_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # get the query, key and value vectors from the in_proj layer's output
        # shape = batch_size, seq_len, embed_dim
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # reshape the query, key, value into inter_shape
        q = q.view(inter_shape).transpose(1,2)
        k = k.view(inter_shape).transpose(1,2)
        v = v.view(inter_shape).transpose(1,2)

        # compute attention scores
        # shape = batch_size, n_heads, seq_len, seq_len
        attn_scores = q @ k.transpose(-1,-2)

        # apply causal mask if required
        if causal_mask:
            mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
            attn_scores.masked_fill_(mask, -torch.inf)
        
        # normalization of attention scores
        attn_scores = attn_scores / (self.d_head ** 0.5)

        # computation of probabilities
        attn_scores = F.softmax(attn_scores, dim = -1)
        
        # computation of output
        # (b, n_heads, seq_len, seq_len) @ (b, n_heads, seq_len, d_head) -> (b, n_heads, seq_len, d_head)
        output = attn_scores @ v

        # transpose and reshape to input shape
        # (b, n_heads, seq_len, d_head) -> (b, seq_len, embed_dim)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)

        # apply out_proj layer on the output tensor
        output = self.out_proj(output)

        return output

class crossAttention(nn.Module):
    def __init__(self,embed_dim : int, num_heads : int, d_cross : int, 
                 in_proj_bias : bool = True, out_proj_bias : bool = True):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, embed_dim, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, embed_dim, bias=in_proj_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        self.n_heads = num_heads
        self.d_head = embed_dim // num_heads


    def forward(self,x, y):
        # x : batch_size, height*width, embed_dim
        # y : batch_size, seq_len, d_cross
        # x.shape -> x.shape
        # query vector from x and key, value from y
        q = self.q_proj(x)
        # y.shape -> x.shape
        k = self.k_proj(y)
        # y.shape -> x.shape
        v = self.v_proj(y)

        b, seq_len, embed_dim = x.shape

        # b, seq_len, embed_dim -> b, seq_len, n_heads, d_head -> b, n_heads, seq_len, d_head
        q = q.view((b, seq_len, self.n_heads, self.d_head)).transpose(1,2)
        k = k.view((b, seq_len, self.n_heads, self.d_head)).transpose(1,2)
        v = v.view((b, seq_len, self.n_heads, self.d_head)).transpose(1,2)

        # b, n_heads, seq_len, d_head @ b, n_heads, d_head, seq_len -> b, n_heads, seq_len, seq_len
        attention_scores = q @ k.transpose(-1,-2)

        attention_scores = attention_scores / math.sqrt(self.d_head)

        attention_scores = F.softmax(attention_scores, dim=-1)
        # b, n_heads, seq_len, d_head
        output = attention_scores @ v
        # b, seq_len, n_heads, d_head
        output = output.transpose(1,2).contiguous()
        # b, seq_len, embed_dim
        output = output.view((b, seq_len, embed_dim))
        # b, seq_len, embed_dim
        output = self.out_proj(output)

        return output


        




    
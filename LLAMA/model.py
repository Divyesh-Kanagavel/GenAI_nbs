import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass
# @dataclass generates boiler plate code for the class including constructor
# __repr__, __str__ etc
@dataclass
class Model_args:
    dim : int = 4096
    n_layers : int = 32
    n_heads : int = 32 # number of heads of the queries
    num_kv_heads : Optional[int] = None # number of heads for keys and values
    vocab_size : int = -1 # will be set from the tokenizer

    multiple_of : int = 256
    ffn_multiplier : Optional[float] = None # used in feedforward layer
    norm_eps : float = 1e-6

    # needed fo kv cache

    max_batch_size : int = 32
    max_seq_len : int = 2048
    device : str = None

# precomputation of the theta and pos frequencies required for rotary position encodings

def precompute_theta_pos_frequencies(head_dim : int, seq_len : int, device : str, theta : float = 10000.0):
    # in the rotatary position encoding , the requurement is that the head dim is divisible by 2
    assert head_dim % 2 == 0 , "head_dim is divisible by 2"
    # theta is computed as 10000 ^ (-2(i-1)/d) where i is the index in embedding table and goes from 1...d/2
    # shape = (head_dim/2)
    theta_list = torch.arange(0, head_dim, 2).float()
    # shape = (head_dim/2)
    theta = 1.0 / (theta ** (theta_list/head_dim)).to(device)

    m = torch.arange(seq_len, device=device) # shape = (seq_len)

    freqs = torch.outer(m, theta) # shape = (seq_len, head_dim/2)

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # shape = (seq_len, head_dim/2)

    return freqs_complex

class RMSNorm(nn.Module):
    def __init__(self, dim : int, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(x : torch.Tensor, n_reps : int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_reps == 1:
        return x
    # (b,seq_len, n_kv_heads, head_dim) -> (b, seq_len,n_kv_heads, 1, head_dim) -> (b, seq_len, n_heads, head_dim)
    return (
    x[:,:,:,None,:].expand(batch_size, seq_len, n_kv_heads,n_reps,head_dim).reshape(
        batch_size, seq_len, n_kv_heads*n_reps, head_dim
    )
    )

class selfAttention(nn.Module):
    def __init__(self, args : Model_args) -> None:
        super().__init__()
        self.args = args
        # number of kv heads
        self.n_kv_heads = args.n_heads if args.num_kv_heads is None else args.num_kv_heads
        self.n_q_heads = args.n_heads
        #ration between q heads and kv heads
        self.n_reps = self.n_q_heads // self.n_kv_heads
        # head dim
        self.head_dim = args.dim // args.n_heads

        # weight matrices
        self.wq = nn.Linear(args.dim, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_q_heads * self.head_dim, args.dim, bias=False)

        # KV cache
        self.k_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads , self.head_dim))
        self.v_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads , self.head_dim))
    
    def forward(self, x:torch.Tensor, freqs_complex : torch.Tensor, start_pos : int):
        batch_size, seq_len, dim = x.shape
        #(B, seq_len, dim) -> (B, seq_len, n_heads, head_dim)
        xq = self.wq(x)
        # (B, seq_len, dim) -> (B, seq_len, num_kv_heads, head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # apply the rotary position encoding to queries and keys
        # (B, seq_len, n_heads, head_dim)
        xq = apply_rotary_pos_encodings(xq, freqs_complex, device=x.device)
        # (B, seq_len, num_kv_heads, head_dim)
        xk = apply_rotary_pos_encodings(xk, freqs_complex, device=x.device)

        # update the kv cache
        self.k_cache[:batch_size, start_pos:start_pos+seq_len] = xk
        self.v_cache[:batch_size, start_pos:start_pos+seq_len] = xv

        #retrive the cached kv values
        #shape = (B, seq_len, n_kv_heads, head_dim)
        keys = self.k_cache[:batch_size, :start_pos+seq_len]
        values = self.v_cache[:batch_size, :start_pos+seq_len]

        # to perform grouped query attention, we have optimized algorithms 
        # which are supported in higher llama models like the 70 b model
        # in the 7b model we use, the GQA is implemented by just creating copies of 
        # the key and value to match the number of query heads
        keys = repeat_kv(keys, self.n_reps)
        values = repeat_kv(values, self.n_reps)

        # proceed like MHA
        # (B, seq_len, n_heads,head_dim) -> (B, n_heads, seq_len, head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(),dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        # (B, n_heads, seq_len, head_dim) -> (B, seq_len, n_heads*head_dim)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args : Model_args) -> None:
        super().__init__()
        self.args = args
        self.dim = args.dim
        # computation of hidden dimension
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_multiplier)
        
        # round hidden dim to a multiple of args.multiple_of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x : torch.Tensor):
        # swiglu activation
        # swiglu(x) = x * sigmoid(beta*x)
        # this way of feeding forward works well but not attributed to any
        # sound theoretical basis
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        out = self.w2(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, args : Model_args) -> None:
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        # kv cache enabled grouped query attention block
        self.attention = selfAttention(self.args)
        # feed forward layer with swiglu activation
        self.feed_forward = FeedForward(self.args)

        # rms norm before attention block
        self.attention_norm = RMSNorm(self.dim, eps = args.norm_eps)
        # rms norm before feed forward layer
        self.ffn_norm = RMSNorm(self.dim, eps = args.norm_eps)
    
    def forward(self, x:torch.Tensor, freqs_complex : torch.Tensor, start_pos : int):
        # Batch_size, seq_len, dim + Batch_size, seq_len, dim -> Batch_size, seq_len, dim
        # attention layer with skip connections
        h = x + self.attention(self.attention_norm(x), freqs_complex,start_pos)
        # feedforward layer
        out = h + self.feed_forward(self.ffn_norm(h))
        # final output token
        return out

# function to apply the rotary position encoding to the input token
def apply_rotary_pos_encodings(x : torch.Tensor, freqs_complex : torch.tensor, device : str) :
    # (Batch_size, seq_len , H, head_dim) -> (Batch_size, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))

    # convert freqs_complex into the shape (Batch_size, seq_len, H, head_dim/2)
    # (seq_len,head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # element wise multiplication of x_complex and freqs_complex to rotate the input token 
    # based on position
    #shape : (Batch_size, seq_len, H, head_dim/2)
    x_complex = x_complex * freqs_complex

    # convert back to the original shape flattening the complex tensor
    x_out = torch.view_as_real(x_complex)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

# the transformer model - meat of the llama 2 architecture

class Transformer(nn.Module):
    def __init__(self, args : Model_args) -> None:
        super().__init__()
        # vocab size should not be the default -1
        assert args.vocab_size!=-1, "Vocab size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)

        # list of layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(DecoderBlock(self.args))
        
        # RMS norm is used instead of the layer norm in transformer paper
        # this is an llama 2 arch specific addition
        self.norm = RMSNorm(self.dim, eps = args.norm_eps)

        # final output layer
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_frequencies(
                              self.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)
        
    def forward(self, tokens:torch.Tensor, start_pos : int):

        batch_size, seq_len = tokens.shape
        # we have kv cache enabled in llama 2, hence one token is taken at a time
        # and the history of tokens is not required every time
        assert seq_len == 1, "Only one token at a time"

        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        for layer in self.layers:
            h = layer(h, freqs_complex,start_pos)
        h = self.norm(h)
        output = self.output(h).float()
        return output

        

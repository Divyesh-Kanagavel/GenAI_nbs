import torch
import torch.nn as nn
import math

# input embedding class

class inputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

# my intuition for multiplication of sqrt(d_model) with the embedding matrix
# there is a division of sqrt(d_model) in the scaled dot product attention which is 
# used to compensate for the increase in variance due to the dot product operation.
# if not done, we will see a heavily skewed distribution of dot product which influences softmax
# the multiplication of sqrt(d_model) with the embedding matrix is done to get the opposite effect
# we want to increase the magnitude to not have a diffuse distribution initially. the embedding matrix provides a standard uniform variance 
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# positional encoding class
class positionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len,dtype=torch.float).unsqueeze(1)
        # denominator of the position encoding formula 
        # PE(pos,2i) = sin(pos/10000^(2i/d_model))
        # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        # denominator = 10000^(2i/d_model)
        # we take a logarithm of the denominator for numerical stability
        denominator = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * denominator)
        pe[:,1::2] = torch.cos(position * denominator)

        # add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)
        # not a learnable parameter but saved as a buffer along with the model
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:,:x.shape[1], :].requires_grad_(False) # no need of gradient computation
        return self.dropout(x)
    

class LayerNorm(nn.Module):
    def __init__(self,eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) # learnable parameter which is multiplied after layernorm to give network the flexibiility to choose the right scale
        self.beta = nn.Parameter(torch.zeros(1)) # learnable paramter which is added
    
    def forward(self, x):
        mean = x.float().mean(dim=-1, keepdim=True)
        std = x.float().std(dim=-1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
# Feed forward block which connects encoder's mha to decoder's mha
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # w1 @ x + b1
        self.linear2 = nn.Linear(d_ff, d_model) # w2 @ x + b2
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# Multihead attention block

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model , n_heads, dropout:float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"
        self.dropout = nn.Dropout(dropout)
        self.dk = self.d_model // self.n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False) # WQ
        self.wk = nn.Linear(d_model, d_model, bias=False) # WK
        self.wv = nn.Linear(d_model, d_model, bias=False) # WV

        self.wo = nn.Linear(d_model, d_model, bias=False) # W0
    
    @staticmethod
    def attention(query, key, value, mask, dropout):
        # query shape : )(batch_size, num_heads, seq_len, dk)
        dk = query.shape[-1]
        # scaled dot product attention
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(dk)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -math.inf)
        # masked softmax 
        attention_scores = attention_scores.softmax(dim = -1)
        # apply dropout if provided
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # return the result qkv and attention scores
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask = None):
        query = self.wq(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.wk(k) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model
        value = self.wv(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model

        # (batch_size, seq_len, d_model) -> (batch_size,seq_len, num_heads, dk) -> (batch_size,num_heads,seq_len, dk, )
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.dk).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.dk).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.dk).transpose(1,2)
        
        x , attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, num_heads, seq_len, dk) -> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.dk * self.n_heads)

        return self.wo(x)

# residual connection block

class ResidualConnection(nn.Module):
    def __init__(self, dropout : float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# encoder block -  a stack of multihead attention blocks and feedforward nn

class EncoderBlock(nn.Module):
    def __init__(self, self_attention : MultiHeadAttention, feed_forward : FeedForwardBlock, dropout : float = 0.2):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    # padding applied to the input of the encdoer to prevent interaction of padding word with the other words.
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers  = layers
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention : MultiHeadAttention, cross_attention : MultiHeadAttention, feed_forward : FeedForwardBlock, dropout : float = 0.2):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    # src_mask : mask for encoder inputs, tgt_mask : mask for decoder inputs
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention(x, enc_out, enc_out, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self,x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = self.projection(x)
        return x


# transformer block which connects encoder, decoder and projection layer
class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed : inputEmbedding,
                  tgt_embed : inputEmbedding, src_pos : positionalEncoding, tgt_pos : positionalEncoding,
                  proj_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self,src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src
    
    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt,enc_out, src_mask, tgt_mask)
        return tgt
    
    def project(self,x):
        return self.proj_layer(x)
    
# function to build the transformer block
def build_transformer(src_vocab_size : int, tgt_vocab_size : int,
                      src_seq_len : int, tgt_seq_len : int, d_model : int = 512,
                      num_blocks : int = 6, n_heads : int = 8, dropout : float = 0.1,
                      d_ff : int = 512, ) -> Transformer:
    src_embed = inputEmbedding(d_model, src_vocab_size)
    tgt_embed = inputEmbedding(d_model, tgt_vocab_size)
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = positionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(num_blocks):
        encoder_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(num_blocks):
        decoder_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,src_pos,tgt_pos,
                               projection_layer)
    # Xavier initialization of model weights
    # W drawn from N ~ (0,2/(n_in + n_out)) to prevent vanishing or exploding gradients
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
    

    




















    
    
    

    
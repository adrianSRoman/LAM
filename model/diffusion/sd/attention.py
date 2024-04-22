import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # NOTE: the rgb channel is the embeding dimension
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # how many embeedings per head

    def forward(self, x, causal_mask=False):
        # x: # (batch, Seq_Len, Dim)

        # (batch, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (batch, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (batch, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (batch, Seq_Len, Dim) -> (batch, Seq_Len, Dim * 3) -> 3 tensor of shape (batch, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch, Seq_Len, Dim) -> (batch, Seq_Len, H, Dim / H) -> (batch, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch, H, Seq_Len, Dim / H) @ (batch, H, Dim / H, Seq_Len) -> (batch, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (batch, H, Seq_Len, Seq_Len) -> (batch, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (batch, H, Seq_Len, Seq_Len) -> (batch, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (batch, H, Seq_Len, Seq_Len) @ (batch, H, Seq_Len, Dim / H) -> (batch, H, Seq_Len, Dim / H)
        output = weight @ v

        # (batch, H, Seq_Len, Dim / H) -> (batch, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 

        # (batch, Seq_Len, H, Dim / H) -> (batch, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (batch, Seq_Len, Dim) -> (batch, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (batch, Seq_Len, Dim)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    # Cross attention makes every pixel watch any token from the prompt and viceversa
    def forward(self, x, y):
        # x (latent): # (batch, Seq_Len_Q, Dim_Q)
        # NOTE: prompt sequence length is 77
        # y (context): # (batch, Seq_Len_KV, Dim_KV) = (batch, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        # Multiply by Wq        
        # (batch, Seq_Len_Q, Dim_Q) -> (batch, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (batch, Seq_Len_KV, Dim_KV) -> (batch, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (batch, Seq_Len_KV, Dim_KV) -> (batch, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (batch, Seq_Len_Q, Dim_Q) -> (batch, Seq_Len_Q, H, Dim_Q / H) -> (batch, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (batch, Seq_Len_KV, Dim_Q) -> (batch, Seq_Len_KV, H, Dim_Q / H) -> (batch, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (batch, Seq_Len_KV, Dim_Q) -> (batch, Seq_Len_KV, H, Dim_Q / H) -> (batch, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (batch, H, Seq_Len_Q, Dim_Q / H) @ (batch, H, Dim_Q / H, Seq_Len_KV) -> (batch, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (batch, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (batch, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (batch, H, Seq_Len_Q, Seq_Len_KV) @ (batch, H, Seq_Len_KV, Dim_Q / H) -> (batch, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (batch, H, Seq_Len_Q, Dim_Q / H) -> (batch, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (batch, Seq_Len_Q, H, Dim_Q / H) -> (batch, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (batch, Seq_Len_Q, Dim_Q) -> (batch, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (batch, Seq_Len_Q, Dim_Q)
        return output

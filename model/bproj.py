import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import math
from trainer.utils import steering_operator

class ReTanh(torch.nn.Module):
    '''
    Rectified Hyperbolic Tangent
    '''
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        beta = self.alpha / torch.tanh(torch.tensor(1.0, dtype=x.dtype))
        return torch.max(torch.tensor(1e-6, dtype=x.dtype), beta * torch.tanh(x))

def positive_kaiming_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        m.weight.data = torch.abs(m.weight.data)  # Ensure weights are positive
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize bias to zero

class BackProjLayer(torch.nn.Module):
    """Spherical Convolutional Neural Netork.
    """

    def __init__(self, Nch=32, tau=None, D=None):
        """Initialization.
        Args:
            Nch (int): number of channels in mic array
            Npx (int): number of pixels in Robinson projection
        """
        super().__init__()
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1] # get number of pixes in tesselation
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.retanh = ReTanh(alpha=1.0)

        self.embed_dim = 484  # Assuming this is the size of your latent vector
        self.num_heads = 4
        self.num_layers = 2
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(self.num_layers):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.05, batch_first=True, dtype=torch.float64))
            self.layer_norm_list.append(nn.LayerNorm(self.embed_dim, dtype=torch.float64))
        # Adding normalization and activation layers before attention
        self.pre_norm = nn.LayerNorm(self.embed_dim, dtype=torch.float64)
        self.pre_activation = nn.ReLU()

        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, dtype=torch.float64)
        self.conv1.apply(positive_kaiming_init)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, dtype=torch.float64)
        self.conv2.apply(positive_kaiming_init)

    def reset_parameters(self):
        std = 1e-5
        self.tau.data.normal_(0, std)
        self.D.data.normal_(0, std)

    def forward(self, S):
        """Vectorized Forward Pass.
        Args:
            S (:obj:`torch.Tensor`): input to be forwarded. (batch_size, Nch, Nch)
        Returns:
            :obj:`torch.Tensor`: output: (batch_size, N_px)
        """
        device = S.device # get the device where the computation is being performed
        self.A = self.A.to(device) # Move A to the appropriate device
        S = S.squeeze(1)
        batch_size, N_ch = S.shape[:2]
        N_px = self.tau.shape[0]
        Ds, Vs = torch.linalg.eigh(S)  # (batch_size, N_ch, N_ch), (batch_size, N_ch, N_ch)
        idx = Ds > 0  # To avoid np.sqrt() issues.
        Ds = torch.where(idx, Ds, torch.zeros_like(Ds)) # apply mask to Ds
        Vs = Vs * torch.sqrt(Ds).unsqueeze(1) # element-wise multiplication between Vs and sqrt(Ds)
        latent_x = torch.matmul(self.D.conj().T, Vs)
        latent_x = torch.linalg.norm(latent_x, dim=2) ** 2 # norm operation along the second dimension and square the result
        
        x = latent_x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        latent_x = x.squeeze(1)
        #x = self.pre_norm(x)
        ##x = self.pre_activation(x)

        #for mhsa_cnt in range(len(self.mhsa_block_list)):
        #    x_attn_in = x 
        #    x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
        #    x = x.squeeze(1) + x_attn_in.squeeze(1)
        #    x = self.layer_norm_list[mhsa_cnt](x)
        #latent_x = x
        
        expanded_A = self.A.unsqueeze(0) # expand to unit in batch dimension
        out = torch.einsum('nij,bjk,nkl->bil', expanded_A, torch.diag_embed(latent_x.cdouble()), expanded_A.transpose(1, 2).conj())
        return out, latent_x

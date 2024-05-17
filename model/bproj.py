import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import math

device = 'cuda:0'

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

class BackProjLayer(torch.nn.Module):
    """Spherical Convolutional Neural Netork.
    """

    def __init__(self, Nch=32, Npx=370, tau=None, D=None):
        """Initialization.
        Args:
            Nch (int): number of channels in mic array
            Npx (int): number of pixels in Robinson projection
        """
        super().__init__()
        self.A = torch.from_numpy(np.load("/scratch/data/repos/LAM/util/steering.npy")).to(device)
        self.A.requires_grad = False
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.retanh = ReTanh(alpha=1.0)
    
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
        S = S.squeeze(1)
        batch_size, N_ch = S.shape[:2]
        if N_ch == 32: # No CDBPN upsampling needed
            N_px = self.tau.shape[0]
            Ds, Vs = torch.linalg.eigh(S)  # (batch_size, N_ch, N_ch), (batch_size, N_ch, N_ch)
            idx = Ds > 0  # To avoid np.sqrt() issues.
            Ds = torch.where(idx, Ds, torch.zeros_like(Ds)) # apply mask to Ds
            Vs = Vs * torch.sqrt(Ds).unsqueeze(1) # element-wise multiplication between Vs and sqrt(Ds)
            latent_x = torch.matmul(self.D.conj().T, Vs)
            latent_x = torch.linalg.norm(latent_x, dim=2) ** 2 # norm operation along the second dimension and square the result
            #latent_x -= self.tau
            #latent_x = F.relu(latent_x) # apply sparcifier operator
            expanded_A = self.A.unsqueeze(0) # expand to unit in batch dimension
            out = torch.einsum('nij,bjk,nkl->bil', expanded_A, torch.diag_embed(latent_x.cdouble()), expanded_A.transpose(1, 2).conj())
        else: # if N_ch == 4
            # perform CDBPN upsampling here
            pass
        return out, latent_x

import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

from trainer.utils import steering_operator

def initialize_scaled_kaiming(layer, scale=1e-6):
    torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data.fill_(1e-6)  # Small bias to avoid dead neurons

class LAM(torch.nn.Module):
    def __init__(self, num_bands=16, Nch=32, tau=None, D=None):
        super(LAM, self).__init__()
        self.num_bands = num_bands
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((self.num_bands, Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((self.num_bands, Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.retanh = nn.ReLU()

        # Convolution layers modified to work across all frequency bands
        self.denoise1 = torch.nn.Conv1d(num_bands, num_bands, kernel_size=3, padding=1, dtype=torch.float64)
        self.denoise2 = torch.nn.Conv1d(num_bands, num_bands, kernel_size=5, padding=2, dtype=torch.float64)
        self.denoise3 = torch.nn.Conv1d(num_bands, num_bands, kernel_size=7, padding=3, dtype=torch.float64)
        self.denoise4 = torch.nn.Conv1d(num_bands, num_bands, kernel_size=9, padding=4, dtype=torch.float64)

        initialize_scaled_kaiming(self.denoise1)
        initialize_scaled_kaiming(self.denoise2)
        initialize_scaled_kaiming(self.denoise3)
        initialize_scaled_kaiming(self.denoise4)

    def reset_parameters(self):
        std = 1e-5  # changes this from 1e-4
        self.tau.data.normal_(0, 1e-7)
        self.D.data.normal_(0, std)

    def forward(self, S):
        device = S.device
        self.A = self.A.to(device)
        batch_size, freq_bands, N_ch = S.shape[:3]
        
        latent_x_list = []
        # encoding (back-projection)
        for i in range(freq_bands):
            # Reshape input for processing
            S_i = S[:, i, :, :]
            Ds, Vs = torch.linalg.eigh(S_i)
            idx = Ds > 0
            Ds = torch.where(idx, Ds, torch.zeros_like(Ds))
            Vs = Vs * torch.sqrt(Ds).unsqueeze(1)

            latent_x = torch.matmul(self.D[i].conj().T, Vs)
            latent_x = torch.linalg.norm(latent_x, dim=2) ** 2
            latent_x -= self.tau[i]
            latent_x_list.append(latent_x)

        # de-noise latent space
        latent_x = torch.stack(latent_x_list, dim=1) # stack all maps from N bands
        latent_x_skip = latent_x.clone() 
        latent_x = self.denoise1(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.denoise2(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.denoise3(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.denoise4(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)

        out_list = []
        # decoding with steering matrix
        for i in range(latent_x.shape[1]):
            latent_i = latent_x[:, i, :]
            out = torch.einsum('nij,bjk,nkl->bil', self.A.unsqueeze(0),
                               torch.diag_embed(latent_i.cdouble()),
                               self.A.unsqueeze(0).transpose(1, 2).conj())
            out_list.append(out)
        out = torch.stack(out_list, dim=1)

        return out, latent_x

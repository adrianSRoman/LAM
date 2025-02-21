import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import math
from trainer.utils import steering_operator

def initialize_scaled_kaiming(layer, scale=1e-6):
    torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data.fill_(1e-6)  # Small bias to avoid dead neurons


class NBProj(nn.Module):
    def __init__(self, num_channels=16):
        super(NBProj, self).__init__()
        self.num_channels = num_channels
        self.bproj_layer = BackProjLayer(num_channels=num_channels)

    def forward(self, S):
        # Apply the updated BackProjLayer to the entire input
        out, x = self.bproj_layer(S)
        return out, x


class BackProjLayer(torch.nn.Module):
    def __init__(self, num_channels=16, Nch=32, tau=None, D=None):
        super().__init__()
        self.num_channels = num_channels
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((self.num_channels, Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((self.num_channels, Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.retanh = nn.ReLU()

        # Convolution layers modified to work across all frequency bands
        self.smooth1 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1, dtype=torch.float64)
        self.smooth2 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2, dtype=torch.float64)
        self.smooth3 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3, dtype=torch.float64)
        self.smooth4 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=9, padding=4, dtype=torch.float64)
        self.smooth5 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=11, padding=5, dtype=torch.float64)
        self.smooth6 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=13, padding=6, dtype=torch.float64)
        self.smooth7 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=15, padding=7, dtype=torch.float64)
        self.smooth8 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=17, padding=8, dtype=torch.float64)

        initialize_scaled_kaiming(self.smooth1)
        initialize_scaled_kaiming(self.smooth2)
        initialize_scaled_kaiming(self.smooth3)
        initialize_scaled_kaiming(self.smooth4)
        initialize_scaled_kaiming(self.smooth5)
        initialize_scaled_kaiming(self.smooth6)
        initialize_scaled_kaiming(self.smooth7)
        initialize_scaled_kaiming(self.smooth8)

    def reset_parameters(self):
        std = 1e-5  # changes this from 1e-4
        self.tau.data.normal_(0, 1e-7)
        self.D.data.normal_(0, std)

    def forward(self, S):
        device = S.device
        self.A = self.A.to(device)
        batch_size, freq_bands, N_ch = S.shape[:3]
        
        latent_x_list = []
        for i in range(freq_bands):
            # Reshape input for processing
            S_i = S[:, i, :, :] #.permute(0, 2, 1, 3).reshape(batch_size * freq_bands, N_ch, -1)
            Ds, Vs = torch.linalg.eigh(S_i)
            idx = Ds > 0
            Ds = torch.where(idx, Ds, torch.zeros_like(Ds))
            Vs = Vs * torch.sqrt(Ds).unsqueeze(1)

            latent_x = torch.matmul(self.D[i].conj().T, Vs)
            latent_x = torch.linalg.norm(latent_x, dim=2) ** 2
            latent_x -= self.tau[i]
            latent_x_list.append(latent_x)

        latent_x = torch.stack(latent_x_list, dim=1) # stack all maps from N bands
        latent_x_skip = latent_x.clone() #.unsqueeze(1)
        latent_x = self.smooth1(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.smooth2(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.smooth3(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.smooth4(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)

        latent_x = self.smooth5(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.smooth6(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.smooth7(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.smooth8(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)

        out_list = []
        for i in range(latent_x.shape[1]):
            latent_i = latent_x[:, i, :]
            out = torch.einsum('nij,bjk,nkl->bil', self.A.unsqueeze(0),
                               torch.diag_embed(latent_i.cdouble()),
                               self.A.unsqueeze(0).transpose(1, 2).conj())
            out_list.append(out)
        out = torch.stack(out_list, dim=1)

        return out, latent_x

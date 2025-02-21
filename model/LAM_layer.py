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


class BackProjLayer(torch.nn.Module):
    def __init__(self, Nch=32, tau=None, D=None):
        super().__init__()
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.relu = nn.ReLU()

        # layers used for cascaded conv with small bias
        self.denoise1 = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1, dtype=torch.float64)
        self.denoise2 = torch.nn.Conv1d(1, 1, kernel_size=5, padding=2, dtype=torch.float64)
        self.denoise3 = torch.nn.Conv1d(1, 1, kernel_size=7, padding=3, dtype=torch.float64)
        self.denoise4 = torch.nn.Conv1d(1, 1, kernel_size=9, padding=4, dtype=torch.float64)
        # He initializtion with small variance to handle small numbers better
        initialize_scaled_kaiming(self.denoise1)
        initialize_scaled_kaiming(self.denoise2)
        initialize_scaled_kaiming(self.denoise3)
        initialize_scaled_kaiming(self.denoise4)

    def reset_parameters(self):
        std = 1e-5 # changes this from 1e-4
        self.tau.data.normal_(0, 1e-7)
        self.D.data.normal_(0, std)

    def forward(self, S):
        device = S.device
        self.A = self.A.to(device)
        S = S.squeeze(1)
        batch_size, N_ch = S.shape[:2]
        Ds, Vs = torch.linalg.eigh(S)
        idx = Ds > 0
        Ds = torch.where(idx, Ds, torch.zeros_like(Ds))
        Vs = Vs * torch.sqrt(Ds).unsqueeze(1)
        latent_x = torch.matmul(self.D.conj().T, Vs)
        latent_x = torch.linalg.norm(latent_x, dim=2) ** 2
        latent_x -= self.tau
        latent_x = latent_x.unsqueeze(1)
        latent_x_skip = latent_x
        latent_s1 = self.denoise1(latent_x)
        latent_x  = latent_x_skip.add(latent_s1)
        latent_x  = self.relu(latent_x)
        latent_s2 = self.denoise2(latent_x)
        latent_x  = latent_x_skip.add(latent_s2)
        latent_x  = self.relu(latent_x)
        latent_s3 = self.denoise3(latent_x)
        latent_x  = latent_x_skip.add(latent_s3)
        latent_x  = self.relu(latent_x)
        latent_s4 = self.denoise4(latent_x)
        latent_x  = latent_x_skip.add(latent_s4)
        latent_x  = self.relu(latent_x)
        latent_x  = latent_x.squeeze(1)
        out = torch.einsum('nij,bjk,nkl->bil', self.A.unsqueeze(0), 
                         torch.diag_embed(latent_x.cdouble()), 
                         self.A.unsqueeze(0).transpose(1, 2).conj())
        return out, latent_x

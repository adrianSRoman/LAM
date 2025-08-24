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
        std = 1e-5
        self.tau.data.normal_(0, 1e-7)
        self.D.data.normal_(0, std)
    
    def forward(self, S):
        device = S.device
        self.A = self.A.to(device)
        batch_size, freq_bands, N_ch = S.shape[:3]
        
        # Vectorized encoding (back-projection) - process all bands at once
        # S shape: [batch_size, freq_bands, N_ch, N_ch]
        # Compute eigendecomposition for all frequency bands simultaneously
        Ds, Vs = torch.linalg.eigh(S)  # Shape: [batch_size, freq_bands, N_ch], [batch_size, freq_bands, N_ch, N_ch]
        
        # Apply threshold and sqrt operation vectorized
        idx = Ds > 0
        Ds = torch.where(idx, Ds, torch.zeros_like(Ds))
        Vs = Vs * torch.sqrt(Ds).unsqueeze(-1)  # Broadcasting across last dimension
        
        # Vectorized matrix multiplication across all bands
        # self.D shape: [num_bands, Nch, Npx]
        # Vs shape: [batch_size, freq_bands, N_ch, N_ch]
        # We want: D[i].conj().T @ Vs[:, i, :, :] for all i
        D_conj_T = self.D.conj().transpose(-2, -1)  # [num_bands, Npx, Nch]
        
        # Batch matrix multiply: [batch_size, freq_bands, Npx, N_ch] @ [batch_size, freq_bands, N_ch, N_ch]
        # -> [batch_size, freq_bands, Npx, N_ch]
        latent_x = torch.matmul(D_conj_T.unsqueeze(0), Vs)
        
        # Compute norm squared across the last dimension
        latent_x = torch.linalg.norm(latent_x, dim=-1) ** 2  # [batch_size, freq_bands, Npx]
        
        # Subtract tau (broadcasting)
        latent_x = latent_x - self.tau.unsqueeze(0)  # [batch_size, freq_bands, Npx]
        
        # Denoising in latent space (same as before)
        latent_x_skip = latent_x.clone()
        latent_x = self.denoise1(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.denoise2(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.denoise3(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        latent_x = self.denoise4(latent_x) + latent_x_skip
        latent_x = self.retanh(latent_x)
        
        # Vectorized decoding with steering matrix - process all bands at once
        # latent_x shape: [batch_size, freq_bands, Npx]
        # Convert to complex and create diagonal matrices
        latent_diag = torch.diag_embed(latent_x.cdouble())  # [batch_size, freq_bands, Npx, Npx]
        
        # self.A shape: [N_ch, Npx]
        # We want: A @ diag(latent_x[i]) @ A.H for all i
        A_expanded = self.A.unsqueeze(0).unsqueeze(0)  # [1, 1, N_ch, Npx]
        A_H_expanded = self.A.unsqueeze(0).unsqueeze(0).transpose(-2, -1).conj()  # [1, 1, Npx, N_ch]
        
        # Batch matrix multiply: [1, 1, N_ch, Npx] @ [batch_size, freq_bands, Npx, Npx] @ [1, 1, Npx, N_ch]
        # -> [batch_size, freq_bands, N_ch, N_ch]
        out = torch.matmul(torch.matmul(A_expanded, latent_diag), A_H_expanded)
        
        return out, latent_x

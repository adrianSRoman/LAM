import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = 'cuda:0'

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.A = torch.from_numpy(np.load("/scratch/data/repos/LAM/util/steering.npy")).to(device)
        self.A.requires_grad = False
        # separate branches for real and imaginary parts
        self.conv1_real = nn.Conv2d(1, 16, kernel_size=3, padding=1, groups=1)  # Process real part
        self.conv1_imag = nn.Conv2d(1, 16, kernel_size=3, padding=1, groups=1)  # Process imaginary part
        self.conv2_real = nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=1)
        self.conv2_imag = nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=1)
        # common layers for both branches
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64)
        self.fc1 = nn.Linear(128 * 8 * 8, 370)

    def forward(self, x):

        real_part = torch.real(x)
        imag_part = torch.imag(x)
        
        # real part
        real_out = F.relu(self.conv1_real(real_part))
        real_out = self.pool(real_out)
        real_out = F.relu(self.conv2_real(real_out))

        # imaginary part
        imag_out = F.relu(self.conv1_imag(imag_part))
        imag_out = self.pool(imag_out)
        imag_out = F.relu(self.conv2_imag(imag_out))
        
        # concatenate real and imag -> conv and pool
        x = torch.cat((real_out, imag_out), dim=1)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        #print("shape", x.shape)
        x = F.relu(self.fc1(x))

        latent_x = x.type(torch.complex64)        
        x = torch.diag_embed(latent_x).to(torch.complex128)
        x = torch.matmul(x, self.A.conj().transpose(0, 1))
        out = torch.matmul(self.A, x)
        return out, latent_x

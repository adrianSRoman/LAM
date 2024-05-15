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
        self.fc_real = nn.Linear(256*32, 370*32)
        self.fc_imag = nn.Linear(256*32, 370*32)
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
        real_out = real_out.reshape(16, -1)
        real_out = self.fc_real(real_out)
        real_out = real_out.view(16, 32, 370)
        # imaginary part
        imag_out = F.relu(self.conv1_imag(imag_part))
        imag_out = self.pool(imag_out)
        imag_out = F.relu(self.conv2_imag(imag_out))
        imag_out = imag_out.reshape(16, -1)
        imag_out = self.fc_imag(imag_out)
        imag_out = imag_out.view(16, 32, 370)
        # concatenate real and imag -> conv and pool
        cpx_out = torch.complex(real_out, imag_out)
        print("cpx_out", cpx_out.shape)
        Ds, Vs = torch.linalg.eigh(x)  # (batch_size, N_ch, N_ch), (batch_size, N_ch, N_ch)
        idx = Ds > 0  # To avoid np.sqrt() issues.
        Ds = torch.where(idx, Ds, torch.zeros_like(Ds)) # apply mask to Ds
        Vs = Vs * torch.sqrt(Ds).unsqueeze(1) # element-wise multiplication between Vs and sqrt(Ds)
        latent_x = torch.matmul(cpx_out.conj().T, Vs)
        print(latent_x.shape)
        latent_x = torch.linalg.norm(latent_x, dim=2) ** 2 # norm operation along the second dimension and square the result
        latent_x -= self.tau
        latent_x = self.retanh(latent_x) # apply sparcifier operator
        latent_x = F.relu(self.conv1(latent_x))
        expanded_A = self.A.unsqueeze(0) # expand to unit in batch dimension
        out = torch.einsum('nij,bjk,nkl->bil', expanded_A, torch.diag_embed(latent_x.cdouble()), expanded_A.transpose(1, 2).conj())
        return out, latent_x

        x = torch.cat((real_out, imag_out), dim=1)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        #print("shape", x.shape)
        x = F.relu(self.fc1(x))

        latent_x = x.type(torch.complex64)        
        x = torch.diag_embed(latent_x).to(torch.complex128)
        # Q @ torch.diag(L.cdouble()) @ Q.T.conj()
        x = torch.matmul(x, self.A.conj().transpose(0, 1))
        out = torch.matmul(self.A, x)
        return out, latent_x

# Assuming you have already defined your device (e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create an instance of EncoderCNN
encoder = EncoderCNN().to(device)

# Generate a random complex tensor of shape (batch_size, channels, height, width)
# You can adjust the dimensions as needed
batch_size = 16
channels = 1
height = 32
width = 32
random_complex_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.complex64, device=device)

# Run the random complex tensor through the network
output, latent_x = encoder(random_complex_tensor)


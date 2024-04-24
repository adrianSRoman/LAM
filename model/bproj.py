import torch
import numpy as np
from torch import nn

device = 'cuda:0'

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
        self.Nch = Nch
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)

    def reset_parameters(self):
        std = 1e-5
        self.tau.data.normal_(0, std)
        self.D.data.normal_(0, std)

    def forward(self, S):
        """Forward Pass.
        Args:
            S (:obj:`torch.Tensor`): input to be forwarded. (N_sample, Npx)
        Returns:
            :obj:`torch.Tensor`: output: (N_sample, Npx)
        """
        N_sample, N_px = S.shape[0], self.tau.shape[0]
        latent_x = torch.zeros((N_sample, N_px)).to(device)
        out = torch.zeros((N_sample, self.Nch, self.Nch), dtype=torch.complex128).to(device)
        S = S.squeeze(1)
        for i in range(N_sample): # Loop to handle linalg.eigh: broadcasting can be slower
            Ds, Vs = torch.linalg.eigh(S[i]) # (Nch, Nch), (Nch, Nch)
            idx = Ds > 0  # To avoid np.sqrt() issues.
            Ds, Vs = Ds[idx], Vs[:, idx]
            latent_x[i] = torch.linalg.norm(self.D.conj().T @ (Vs * torch.sqrt(Ds)), axis=1) ** 2 # (Npx, Nch) dot ((Nch, Nch) * (Nch, Nch))
            latent_x[i] -= self.tau
            #print("shape of out", (self.A @ torch.diag(latent_x[i].cdouble()) @ self.A.T.conj()).dtype)
            out[i] = self.A @ torch.diag(latent_x[i].cdouble()) @ self.A.T.conj()
        #print(out.dtype, latent_x.dtype)
        return out, latent_x


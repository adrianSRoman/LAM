import torch
import torch.nn as nn
from trainer.utils import get_field

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, target, pred):
        # Ensure both target and pred are complex tensors
        assert target.is_complex() and pred.is_complex(), "Both tensors must be complex"

        # Separate real and imaginary parts
        target_real = target.real
        target_imag = target.imag
        pred_real = pred.real
        pred_imag = pred.imag

        # Compute MSE loss for real and imaginary parts
        mse_loss_real = self.mse_loss(target_real, pred_real)
        mse_loss_imag = self.mse_loss(target_imag, pred_imag)

        # Combine the losses
        total_loss = mse_loss_real + mse_loss_imag

        return total_loss

class l1_reg_loss(nn.Module):
    def __init__(self, N_max=20, device='cuda:0'):
        super(l1_reg_loss, self).__init__()
        self.R_xyz = torch.from_numpy(get_field()).to(device)
        self.N_max = 20
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, target, pred, latent):
        l1_loss = self.l1_loss(target, pred)
        # Sort the latent tensor and get indices of top N_max values
        #_, sorted_indices = torch.sort(latent, descending=True)
        #max_idx = sorted_indices[:self.N_max]
        # get the coordinates of the N_max values only
        #max_xyz = self.R_xyz[:, max_idx].T
        # parwise ditances
        #dists = torch.nn.functional.pdist(max_xyz, p=2)
        #std_dist = torch.std(dists)
        #total_loss = l1_loss + std_dist
        return l1_loss, 0*l1_loss, 0*l1_loss#std_dist

# Mean Squared Error + Dispersion loss
class MSEDLoss(nn.Module):
    def __init__(self, N_max=20, dispersion_weight=0.01, device='cuda:0'):
        super(MSEDLoss, self).__init__()
        self.R_xyz = torch.from_numpy(get_field()).to(device)
        self.N_max = N_max
        self.mse_loss = ComplexMSELoss()
        self.dispersion_weight = dispersion_weight
        
    def forward(self, target, pred, latent):
        mse_loss = self.mse_loss(target, pred)
        dispersion_loss = self.calculate_dispersion_loss(latent)
        total_loss = mse_loss + self.dispersion_weight * dispersion_loss
        
        return total_loss, mse_loss, self.dispersion_weight * dispersion_loss
    
    def calculate_dispersion_loss(self, latent):
        max_indices = torch.topk(latent, self.N_max).indices
        max_xyz = self.R_xyz[:, max_indices].T
        
        centered_xyz = max_xyz - max_xyz.mean(dim=0)
        cov_matrix = torch.matmul(centered_xyz.T, centered_xyz) / (self.N_max - 1)
        e_vals = torch.linalg.eigvalsh(cov_matrix)
        
        return e_vals.sum()

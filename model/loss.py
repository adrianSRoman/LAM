import torch
import torch.nn as nn
from trainer.utils import get_field

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

class l1_reg_loss(nn.Module):
    def __init__(self):
        super(l1_reg_loss, self).__init__()
        self.R_xyz = torch.from_numpy(get_field()).to('cuda:0')
        self.N_max = 20
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, target, pred, latent):
        l1_loss = self.l1_loss(target, pred)
        # Sort the latent tensor and get indices of top N_max values
        _, sorted_indices = torch.sort(latent, descending=True)
        max_idx = sorted_indices[:self.N_max]
        # get the coordinates of the N_max values only
        max_xyz = self.R_xyz[:, max_idx].T
        # parwise ditances
        dists = torch.nn.functional.pdist(max_xyz, p=2)
        std_dist = torch.std(dists)
        total_loss = l1_loss + std_dist
        return total_loss, std_dist

class l1_cov_loss(nn.Module):
    def __init__(self, N_max=20):
        super(l1_cov_loss, self).__init__()
        self.R_xyz = torch.from_numpy(get_field()).to('cuda:0')
        self.N_max = N_max
        print("Max value being loaded", N_max)
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, target, pred, latent):
        l1_loss = self.l1_loss(target, pred)
        # Sort the latent tensor and get indices of top N_max values
        _, sorted_indices = torch.sort(latent, descending=True)
        max_idx = sorted_indices[:self.N_max]
        # get the coordinates of the N_max values only
        max_xyz = self.R_xyz[:, max_idx].T
        # get mean coordinate
        mean_xyz = max_xyz.mean(dim=0)
        centered_xyz = max_xyz - mean_xyz
        cov_matrix = torch.matmul(centered_xyz.t(), centered_xyz) / (max_xyz.size(0) - 1)
        e_vals, e_vecs = torch.linalg.eigh(cov_matrix)
        e_val_sum = torch.sum(e_vals)
        total_loss = l1_loss + 0.02*e_val_sum
        return total_loss, l1_loss, 0.02*e_val_sum

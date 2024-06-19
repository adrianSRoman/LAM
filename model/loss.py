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
        print("initialized R", self.R_xyz.shape, self.R_xyz[0].shape)
        self.N_max = 12
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, target, pred, latent):
        l1_loss = self.l1_loss(target, pred)
        # Sort the latent tensor and get indices of top N_max values
        _, sorted_indices = torch.sort(latent, descending=True)
        max_idx = sorted_indices[:self.N_max]
        # get the coordinates of the N_max values only
        max_xyz = self.R_xyz[:, max_idx].T
        # parwise ditances
        avg_dist = torch.nn.functional.pdist(max_xyz, p=2)
        avg_dist = torch.sum(avg_dist)
        total_loss = l1_loss + avg_dist/self.N_max
        return total_loss

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from model.bproj import BackProjLayer

class NBProj(nn.Module):
    def __init__(self, num_channels=9):
        super(NBProj, self).__init__()
        self.bproj_layers = nn.ModuleList([BackProjLayer() for _ in range(num_channels)])

    def forward(self, S):
        out_list = []
        x_list = []
        
        # apply the BackProjLayer to each frequency band
        for i in range(S.size(1)):  # iterating over the freq_band dimension
            out, x = self.bproj_layers[i](S[:, i, :, :])  # process each freq_band separately
            out_list.append(out)
            x_list.append(x)
        
        # stack outputs along the freq_band dimension
        out = torch.stack(out_list, dim=1)
        x = torch.stack(x_list, dim=1)
        return out, x

import torch
import numpy as np
from torch import nn

from model.nbproj import NBProj
from model.cdbpn import Net as CDBPN

class CDBPNProj(nn.Module):
    def __init__(self, num_channels=9, base_filter=32, feat=128, num_stages=10, scale_factor=8):
        super(CDBPNProj, self).__init__()
        self.cdbpn = CDBPN(num_channels, base_filter,  feat, num_stages, scale_factor=scale_factor)
        self.nbproj = NBProj(num_channels) 

    def forward(self, S):
        S_pred = self.cdbpn(S.real, S.imag) # perform upsampling
        out, x = self.nbproj(S_pred)        # perform series of bprojs 
        return out, x


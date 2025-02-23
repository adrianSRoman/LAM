import torch
import numpy as np
from torch import nn

from model.LAM import LAM
from model.cdbpn import Net as CDBPN

class UpLAM(nn.Module):
    def __init__(self, num_bands=16, base_filter=32, feat=128, num_stages=10, scale_factor=8):
        super(UpLAM, self).__init__()
        self.cdbpn = CDBPN(num_bands, base_filter,  feat, num_stages, scale_factor=scale_factor)
        self.lam = LAM(num_bands) 

    def forward(self, S):
        S_pred = self.cdbpn(S.real, S.imag) # perform upsampling
        out, x = self.lam(S_pred)        # perform series of bprojs 
        return out, x


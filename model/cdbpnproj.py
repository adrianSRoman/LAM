import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from bproj import BackProjLayer
from cdbpn import Net as CDBPN

class CDBPNProj(nn.Module):
    def __init__(self, num_channels=1, base_filter=32, feat=128, num_stages=10, scale_factor=8):
        super(CDBPNProj, self).__init__()
        self.cdbpn = CDBPN(num_channels, base_filter,  feat, num_stages, scale_factor=scale_factor)
        self.bproj = BackProjLayer()

    def forward(self, S):
        S_pred = self.cdbpn(S.real, S.imag)      # perform upsampling
        out, x = self.bproj(S_pred) # get prediction and latent
        return out, x

# Code to setup mixed precision and perform model profiling
"""
# Ensure model and data are on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CDBPNProj().to(device)
input_tensor = torch.randn((8, 1, 4, 4), device=device, dtype=torch.complex128)

from torch.cuda.amp import autocast, GradScaler
# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
# Mixed precision setup
scaler = GradScaler()

# Define a function to print profiling results to stdout
def trace_handler(p):
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    p.export_chrome_trace("./log/trace_" + str(p.step_num) + ".json")

# Basic profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=trace_handler,
    record_shapes=True,
    with_stack=True
) as p:
    num_steps = 10
    for step in range(num_steps):
        with autocast():
            output = model(input_tensor)
        p.step()
"""

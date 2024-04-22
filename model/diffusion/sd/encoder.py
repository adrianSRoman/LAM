import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch, Channel, Height, Width) -> (batch, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
             # (batch, 128, Height, Width) -> (batch, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (batch, 128, Height, Width) -> (batch, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (batch, 128, Height, Width) -> (batch, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (batch, 128, Height / 2, Width / 2) -> (batch, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256), 
            
            # (batch, 256, Height / 2, Width / 2) -> (batch, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (batch, 256, Height / 2, Width / 2) -> (batch, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # (batch, 256, Height / 4, Width / 4) -> (batch, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512), 
            
            # (batch, 512, Height / 4, Width / 4) -> (batch, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (batch, 512, Height / 4, Width / 4) -> (batch, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (batch, 512, Height / 8, Width / 8) -> (batch, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (batch, 512, Height / 8, Width / 8) -> (batch, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (batch, 512, Height / 8, Width / 8 -> (batch, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (batch, 512, Height / 8, Width / 8) -> (batch, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (batch, 512, Height / 8, Width / 8) -> (batch, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (batch, 512, Height / 8, Width / 8) -> (batch, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (batch, 512, Height / 8, Width / 8) -> (batch, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (batch, 512, Height / 8, Width / 8) -> (batch, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (batch, 8, Height / 8, Width / 8) -> (batch, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )


    def forward(self, x, noise):
        # x: (batch, Channel, Height, Width)
        # noise: (batch, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (batch, Channel, Height, Width) -> (batch, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (batch, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (batch, 8, Height / 8, Width / 8) -> two tensors of shape (batch, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (batch, 4, Height / 8, Width / 8) -> (batch, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (batch, 4, Height / 8, Width / 8) -> (batch, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (batch, 4, Height / 8, Width / 8) -> (batch, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (batch, 4, Height / 8, Width / 8) -> (batch, 4, Height / 8, Width / 8)
        # sample from the normal distribution
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x

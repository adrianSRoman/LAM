import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np
from torchsummary import summary

device = 'cuda:0'

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18Latent(nn.Module):
    def __init__(self, mode=None, num_classes=370):
        super(ResNet18Latent, self).__init__()
        self.A = torch.from_numpy(np.load("/scratch/data/repos/LAM/util/steering.npy")).to(device)
        self.A.requires_grad = False
        
        # Input feature mode
        self.mode = mode

        self.in_planes = 64
        self.conv1_re = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_re = nn.BatchNorm2d(64)
        self.layer1_re = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2_re = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3_re = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4_re = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear_re = nn.Linear(512*BasicBlock.expansion, 2*num_classes)
    
        self.in_planes = 64
        self.conv1_im = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_im = nn.BatchNorm2d(64)
        self.layer1_im = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2_im = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3_im = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4_im = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear_im = nn.Linear(512*BasicBlock.expansion, 2*num_classes)
        
        self.linear_out1 = nn.Linear(4*num_classes, 3*num_classes)
        self.linear_out2 = nn.Linear(3*num_classes, 2*num_classes)
        self.linear_out3 = nn.Linear(2*num_classes, num_classes)

        # use He initialization for ReLu-based network
        self.apply(initialize_weights)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x_re, x_im = torch.real(x), torch.imag(x)
        
        # real
        out_re = F.relu(self.bn1_re(self.conv1_re(x_re)))
        out_re = self.layer1_re(out_re)
        out_re = self.layer2_re(out_re)
        out_re = self.layer3_re(out_re)
        out_re = self.layer4_re(out_re)
        out_re = F.max_pool2d(out_re, 4)
        out_re = out_re.view(out_re.size(0), -1)
        out_re = self.linear_re(out_re)
        # imag
        out_im = F.relu(self.bn1_im(self.conv1_im(x_im)))
        out_im = self.layer1_im(out_im)
        out_im = self.layer2_im(out_im)
        out_im = self.layer3_im(out_im)
        out_im = self.layer4_im(out_im)
        out_im = F.max_pool2d(out_im, 4)
        out_im = out_im.view(out_im.size(0), -1)
        out_im = self.linear_im(out_im)
    
        out_reim = torch.cat((out_re, out_im), dim=-1)
        latent_x = self.linear_out1(out_reim)
        latent_x = self.linear_out2(latent_x)
        latent_x = F.relu(self.linear_out3(latent_x))
        latent_x = latent_x.type(torch.complex64)
        
        x = torch.diag_embed(latent_x).to(torch.complex128)
        x = torch.matmul(x, self.A.conj().transpose(0, 1))
        out = torch.matmul(self.A, x)
        return out


#model = ResNet18Latent()
#summary(model, (9, 32, 32)) 
#dummy_input = torch.randn(1, 9, 32, 32, dtype=torch.cfloat)

#out = model(dummy_input)

#print(dummy_input.shape)

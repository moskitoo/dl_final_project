import os
import numpy as np
import glob
import PIL.Image as Image


# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
#from torchsummary import summary
import torch.optim as optim
from torchvision.transforms import v2
from time import time
import wandb

wandb.login()

class BaseUnet(nn.Module):
    def __init__(self, num_inputs=11):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(num_inputs, 32, 3, padding=1)
        self.downconv0 = nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1)  # 128 -> 64
        self.batchnorm0 = nn.BatchNorm2d(32)

        self.enc_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.downconv1 = nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1)   # 64 -> 32
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.downconv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, dilation=1)   # 32 -> 16
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.enc_conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.downconv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1, dilation=1)   # 16 -> 8
        self.batchnorm3 = nn.BatchNorm2d(128)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(128, 128, 3, padding=1)

        # decoder (upsampling)
        self.upconv0 = nn.ConvTranspose2d(128, 128, 2, stride=2, padding= 0, output_padding=0, dilation=1)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(256, 64, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding= 0, output_padding=0, dilation=1)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 32, 3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(32)

        self.upconv2 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding= 0, output_padding=0, dilation=1)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(32)

        self.upconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding= 0, output_padding=0, dilation=1)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x):
        # encoder
        e0_skip = F.relu(self.enc_conv0(x))
        e0_skip = self.batchnorm0(e0_skip)
        e0 = self.downconv0(e0_skip)
        e1_skip = F.relu(self.enc_conv1(e0))
        e1_skip = self.batchnorm1(e1_skip)
        e1 = self.downconv1(e1_skip)
        e2_skip = F.relu(self.enc_conv2(e1))
        e2_skip = self.batchnorm2(e2_skip)
        e2 = self.downconv2(e2_skip)
        e3_skip = F.relu(self.enc_conv3(e2))
        e3_skip = self.batchnorm3(e3_skip)
        e3 = self.downconv3(e3_skip)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        # print(f'e0.skip.shape: {e0_skip.shape}')
        # print(f'e1.skip.shape: {e1_skip.shape}')
        # print(f'e2.skip.shape: {e2_skip.shape}')
        # print(f'e3.skip.shape: {e3_skip.shape}')
        # print(f'b.shape: {self.upconv0(b).shape}')
        b = torch.cat([e3_skip, self.upconv0(b)], 1)

        # decoder
        d0 = F.relu(self.dec_conv0(b))
        d0 = self.batchnorm4(d0)
        d0 = torch.cat([e2_skip, self.upconv1(d0)], 1)
        d1 = F.relu(self.dec_conv1(d0))
        d1 = self.batchnorm5(d1)
        d1 = torch.cat([e1_skip, self.upconv2(d1)], 1)
        d2 = F.relu(self.dec_conv2(d1))
        d2 = self.batchnorm6(d2)
        d2 = torch.cat([e0_skip, self.upconv3(d2)], 1)
        d3 = self.dec_conv3(d2)  # no activation
        return d3.squeeze(1)

class DoubleUnet(nn.Module):
    def __init__(self, num_inputs1=11, num_inputs2=12):
        super().__init__()
        self.unet1 = BaseUnet(num_inputs1)
        self.unet2 = BaseUnet(num_inputs2)

    def forward(self, x):
        x1 = self.unet1(x)

        x1 = torch.cat([x, x1], 1)
        x2 = self.unet2(x1)
       
        return x2
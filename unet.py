import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(11, 64, 3, padding=1)  # Changed from 3 to 11
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        e0_skip = F.relu(self.enc_conv0(x))  # (batch_size, 64, 512, 512)
        e0 = self.pool0(e0_skip)            # (batch_size, 64, 256, 256)
        e1_skip = F.relu(self.enc_conv1(e0))  # (batch_size, 64, 256, 256)
        e1 = self.pool1(e1_skip)            # (batch_size, 64, 128, 128)
        e2_skip = F.relu(self.enc_conv2(e1))  # (batch_size, 64, 128, 128)
        e2 = self.pool2(e2_skip)            # (batch_size, 64, 64, 64)
        e3_skip = F.relu(self.enc_conv3(e2))  # (batch_size, 64, 64, 64)
        e3 = self.pool3(e3_skip)            # (batch_size, 64, 32, 32)

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))  # (batch_size, 64, 32, 32)

        # Decoder
        d0 = torch.cat([e3_skip, F.interpolate(b, size=e3_skip.shape[2:], mode='bilinear', align_corners=False)], 1)
        d0 = F.relu(self.dec_conv0(d0))  # (batch_size, 64, 64, 64)

        d1 = torch.cat([e2_skip, F.interpolate(d0, size=e2_skip.shape[2:], mode='bilinear', align_corners=False)], 1)
        d1 = F.relu(self.dec_conv1(d1))  # (batch_size, 64, 128, 128)

        d2 = torch.cat([e1_skip, F.interpolate(d1, size=e1_skip.shape[2:], mode='bilinear', align_corners=False)], 1)
        d2 = F.relu(self.dec_conv2(d2))  # (batch_size, 64, 256, 256)

        d3 = torch.cat([e0_skip, F.interpolate(d2, size=e0_skip.shape[2:], mode='bilinear', align_corners=False)], 1)
        d3 = self.dec_conv3(d3)  # (batch_size, 1, 512, 512)

        # Upsample final output to match input size
        output = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)  # Match label size (1024, 1024)

        return output



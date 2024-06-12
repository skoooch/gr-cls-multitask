"""
Original Comments from steven:
            This file contains the architecture with the best performance for
            both CLS and Grasp tasks. It is the AlexnetMap_v5 model in the 
            alexnet_old.py file.

            This model using the first two pretrained layers of Alexnet and 
            outputs a map of predictions.

This file is the multi tasking network with the Alexnet 
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import copy 

from torchvision.models import alexnet


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class Multi_AlexnetMap_v3(nn.Module):
    def __init__(self):
        super(Multi_AlexnetMap_v3, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = copy.deepcopy(pretrained_alexnet.features[:3])
        self.d_features = copy.deepcopy(pretrained_alexnet.features[:3])
        self.features = nn.Sequential(
            nn.Conv2d(64+64, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            
        )
        self.grasp = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 5, kernel_size=11, stride=4, output_padding=1),
            nn.Tanh()
        )

        self.grasp_confidence = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 5, kernel_size=11, stride=4, output_padding=1),
            nn.Tanh()
        )

        self.cls_confidence = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )
        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        for m in self.grasp.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        for m in self.cls_confidence.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        for m in self.grasp_confidence.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x, is_grasp):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        if is_grasp:
            out = self.grasp(x)
            confidence = self.grasp_confidence(x)
        else:
            out = self.cls(x)
            confidence = self.cls_confidence(x)
        out = torch.cat((out, confidence), dim=1)
        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        
        for param in self.d_features.parameters():
            param.requires_grad = True

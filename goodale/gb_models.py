"""
This file contains a class that extracts a submodel from
a pretrained model, up until the selected layer for
visualization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WidthFeatures(nn.Module):
    """This class extracts the 'feature' module from AlexnetGrasp_v5 model.
        
    The indexing of the rgb / depth feature net is as follows:
        0 - nn.Conv2d(64+64, 32, kernel_size=5, padding=2),
        1 - nn.ReLU(inplace=True),
        2 - nn.Dropout(0.3),
        3 - nn.MaxPool2d(kernel_size=3, stride=2),
        4 - nn.Conv2d(32, 64, kernel_size=3, padding=1),
        5 - nn.ReLU(inplace=True),
        6 - vnn.Dropout(0.3),
        7 - nn.Conv2d(64, 64, kernel_size=3, padding=1),
        8 - nn.ReLU(inplace=True),
        9 - nn.Dropout(0.3),
        10 - nn.Conv2d(64, 64, kernel_size=3, padding=1),
        11 - nn.ReLU(inplace=True),
        12 - nn.Dropout(0.3),

    Specify the visualization layer using the above indexing pattern for
    <output> parameter.
    """
    def __init__(self, model, output):
        super(WidthFeatures, self).__init__()
        self.output = output
        self.model = model
        self.layers = {}
        for n, c in model.hidden.named_children():
            self.layers[n] = c

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.model.rgb_features(rgb)
        d = self.model.d_features(d)
        x = torch.cat((rgb, d), dim=1)
        x = self.model.features(x)
        x = x.flatten(start_dim=1)
        for idx in self.layers:
            x = self.layers[idx](x)
            if idx == self.output:
                return x

        return x


class WidthRgbdFeatures(nn.Module):
    """This class extracts the rgbd feature nets from AlexnetGrasp_v5 model.
    
    Users can specify whether to extract the 'rgb' or the 'depth' feature
    net to extract.
    
    The indexing of the rgb / depth feature net is as follows:
        0 - nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        1 - nn.ReLU(inplace=True),
        2 - nn.MaxPool2d(kernel_size=3, stride=2),
        3 - nn.Conv2d(64, 192, kernel_size=5, padding=2),
        4 - nn.ReLU(inplace=True),
        5 - nn.MaxPool2d(kernel_size=3, stride=2)

    Specify the visualization layer using the above indexing pattern for
    <output> parameter.
    """
    def __init__(self, model, output, feature_type='rgb'):
        super(WidthRgbdFeatures, self).__init__()
        self.output = output
        self.feature_type = feature_type
        self.layers = {}
        if feature_type == 'rgb':
            for n, c in model.rgb_features.named_modules():
                if isinstance(c, nn.Conv2d):
                    self.layers[n] = c
        elif feature_type == 'd':
            for n, c in model.d_features.named_children():
                self.layers[n] = c

        print(self.layers)

    def forward(self, x):
        if self.feature_type == 'rgb':
            x = x[:, :3, :, :]
        elif self.feature_type == 'd':
            x = torch.unsqueeze(x[:, 3, :, :], dim=1)
            x = torch.cat((x, x, x), dim=1)

        for idx in self.layers:
            x = self.layers[idx](x)
            if idx == self.output:
                return x

        return x
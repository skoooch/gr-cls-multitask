import torch 
import torch
import torch.nn.functional as F
import torch.nn as nn
from parameters import Params
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
import os
from torchvision.models import alexnet
params = Params()
import copy 


class Multi_AlexnetMap_Width(nn.Module):
    def __init__(self, weights_path):
        super(Multi_AlexnetMap_Width, self).__init__()
        trained_model =  Multi_AlexnetMap_v3().to('cuda')
        trained_model.load_state_dict(torch.load(weights_path))
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = copy.deepcopy(pretrained_alexnet.features[:3])
        self.rgb_features.load_state_dict(trained_model.rgb_features.state_dict())
        self.d_features = copy.deepcopy(pretrained_alexnet.features[:3])
        self.d_features.load_state_dict(trained_model.d_features.state_dict())
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
        self.features.load_state_dict(trained_model.features.state_dict())
        self.hidden = nn.Sequential(nn.Linear(64*13*13, 4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(),
                                    nn.Linear(4096,4096),
                                    nn.ReLU(inplace=True))
        self.dense = nn.Linear(4096, 1)
        
        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)
        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        
        for layer in self.hidden:
            x = layer(x)
        out = self.dense(x)
        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        
        for param in self.d_features.parameters():
            param.requires_grad = True
def test_batch():
    model_name = params.MODEL_NAME
    weights_dir = params.MODEL_PATH
    weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
    model =  Multi_AlexnetMap_Width(weights_path).to('cuda')
    img = torch.zeros((5,4,224,224)).to('cuda')
    print(model(img).detach())
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
    
    def forward(self, x, is_grasp=True, shap_mask=[], activations=[], dissociate=[], connect = False):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)
        # d = torch.zeros(d.shape).to("cuda")
        # First Layer Shapley!!!!! -------------------------------------------------------
        # if shap_mask != []:
            
        #     rgb_conv_out = self.rgb_features[0](rgb)
        #     shap_mask_idx = shap_mask.bool()  # Convert to a boolean mask for indexing
        #     # Broadcast activations to match the shape of the output
        #     broadcasted_activations = activations.unsqueeze(0)  # Add batch dimension
        #     broadcasted_activations = broadcasted_activations.expand_as(rgb_conv_out)  # Expand to match rgb_conv_out's shape
        #     # Set rgb_conv_out where shap_mask is True
        #     rgb_conv_out[:, shap_mask_idx, :, :] = broadcasted_activations[:, shap_mask_idx, :, :]
        #     rgb = self.rgb_features[1](rgb_conv_out)
        #     rgb = self.rgb_features[2](rgb)
        # else:
        # ---------------------------------------------------------------------------------
        rgb = self.rgb_features[0](rgb)
        rgb = self.rgb_features[1](rgb)
        rgb = self.rgb_features[2](rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)
        if dissociate != []:
            mask_idx = torch.zeros(x.shape[1], dtype=torch.bool).to(x.device)
            mask_idx[dissociate[0]] = True
            x[:,mask_idx,:,:] = 0
        if shap_mask != []:
            shap_mask_idx = shap_mask.bool()  # Convert to a boolean mask for indexing
            # Broadcast activations to match the shape of the output
            broadcasted_activations = activations.unsqueeze(0)  # Add batch dimension
            broadcasted_activations = broadcasted_activations.expand_as(x)  # Expand to match rgb_conv_out's shape
            # Set rgb_conv_out where shap_mask is True
            x[:, shap_mask_idx, :, :] = broadcasted_activations[:, shap_mask_idx, :, :]
            for i in range(len(self.features)):
                x = self.features[i](x)
            #------------shapley non first layer ----------
            # features_conv_out = self.features[0:11](x)
            # shap_mask_idx = shap_mask.bool()  # Convert to a boolean mask for indexing
            # # Broadcast activations to match the shape of the output
            # broadcasted_activations = activations.unsqueeze(0)  # Add batch dimension
            # broadcasted_activations = broadcasted_activations.expand_as(features_conv_out)  # Expand to match rgb_conv_out's shape
            # # Set rgb_conv_out where shap_mask is True
            # features_conv_out[:, shap_mask_idx, :, :] = broadcasted_activations[:, shap_mask_idx, :, :]
            # x = self.features[11:](features_conv_out)
        elif dissociate != []:
            prev = 0
            for index, i in enumerate([1,5,8,11]):
                index +=1
                features_conv_out = self.features[prev:i](x)
                prev = i
                mask_idx = torch.zeros(features_conv_out.shape[1], dtype=torch.bool).to(x.device)
                mask_idx[dissociate[index]] = True
                features_conv_out[:,mask_idx,:,:] = 0
                x = features_conv_out
            x = self.features[11:](x)
        else:
            x = self.features(x)
        if is_grasp:
            out = self.grasp(x)
            confidence = self.grasp_confidence(x)
        else:
            out = x
            for i in range(len(self.cls)):
                out = self.cls[i](out)
            # out = self.cls(x)
            confidence = self.cls_confidence(x)
        out = torch.cat((out, confidence), dim=1)
        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        
        for param in self.d_features.parameters():
            param.requires_grad = True
            
    def forward_with_connection_mask(self, x, connection_mask, avg_conn_activations, layer_i):
        # x: [batch, in_channels, H, W]
        W = self.features[layer_i].weight  # [out_channels, in_channels, kH, kW]
        b = self.features[layer_i].bias

        # Standard convolution
        conv_out = self.features[layer_i](x)

        # Replace masked connections with average activation
        for tgt in range(W.shape[0]):
            for src in range(W.shape[1]):
                if connection_mask[tgt, src]:
                    # Remove actual contribution
                    x_src = x[:, src:src+1, :, :]
                    w = W[tgt:tgt+1, src:src+1, :, :]
                    actual_contrib = torch.nn.functional.conv2d(x_src, w, bias=None,
                                                            stride=self.features[layer_i].stride,
                                                            padding=self.features[layer_i].padding)
                    conv_out[:, tgt] -= actual_contrib.squeeze(1)
                    # Add average activation
                    avg_act = torch.tensor(avg_conn_activations[tgt, src]).to(conv_out.device)
                    conv_out[:, tgt] += avg_act
        return conv_out
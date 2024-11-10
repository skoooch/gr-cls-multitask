# normalize the input image to have appropriate mean and standard deviation as specified by pytorch
import os 
import cv2
from torch import layer_norm, optim
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from training.single_task.evaluation import get_cls_acc, get_grasp_acc, visualize_grasp, visualize_cls
from torchvision.utils import make_grid
params = Params()

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')

# AlexNet with 1st, 2nd layer pretrained on Imagenet
model =  Multi_AlexnetMap_v3().to('cuda')
model.load_state_dict(torch.load(weights_path))
model.eval()
# function to massage img_tensor for using as input to plt.imshow()
def image_converter(im):
    
    # move the image to cpu
    im_copy = im.cpu()
    
    # for plt.imshow() the channel-dimension is the last
    # therefore use transpose to permute axes
    im_copy = denormalize(im_copy.clone().detach()).numpy()
    im_copy = im_copy.transpose(1,2,0)
    
    # clip negative values as plt.imshow() only accepts 
    # floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1) 
    
    return im_copy

import torch.nn as nn
# class to compute image gradients in pytorch
class RGBgradients(nn.Module):
    def __init__(self, weight): # weight is a numpy array
        super().__init__()
        k_height, k_width = weight.shape[1:]
        # assuming that the height and width of the kernel are always odd numbers
        padding_x = int((k_height-1)/2)
        padding_y = int((k_width-1)/2)
        
        # convolutional layer with 3 in_channels and 6 out_channels 
        # the 3 in_channels are the color channels of the image
        # for each in_channel we have 2 out_channels corresponding to the x and the y gradients
        self.conv = nn.Conv2d(3, 6, (k_height, k_width), bias = False, 
                              padding = (padding_x, padding_y) )
        # initialize the weights of the convolutional layer to be the one provided
        # the weights correspond to the x/y filter for the channel in question and zeros for other channels
        weight1x = np.array([weight[0], 
                             np.zeros((k_height, k_width)), 
                             np.zeros((k_height, k_width))]) # x-derivative for 1st in_channel
        
        weight1y = np.array([weight[1], 
                             np.zeros((k_height, k_width)), 
                             np.zeros((k_height, k_width))]) # y-derivative for 1st in_channel
        
        weight2x = np.array([np.zeros((k_height, k_width)),
                             weight[0],
                             np.zeros((k_height, k_width))]) # x-derivative for 2nd in_channel
        
        weight2y = np.array([np.zeros((k_height, k_width)), 
                             weight[1],
                             np.zeros((k_height, k_width))]) # y-derivative for 2nd in_channel
        
        
        weight3x = np.array([np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width)),
                             weight[0]]) # x-derivative for 3rd in_channel
        
        weight3y = np.array([np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width)), 
                             weight[1]]) # y-derivative for 3rd in_channel
        
        weight_final = torch.from_numpy(np.array([weight1x, weight1y, 
weight2x, weight2y,
weight3x, weight3y])).type(torch.FloatTensor)
        
        if self.conv.weight.shape == weight_final.shape:
            self.conv.weight = nn.Parameter(weight_final)
            self.conv.weight.requires_grad_(False)
        else:
            print('Error: The shape of the given weights is not correct')
    
    # Note that a second way to define the conv. layer here would be to pass group = 3 when calling torch.nn.Conv2d
    
    def forward(self, x):
        return self.conv(x)
    
def grad_loss(img, beta = 1, device = 'cpu'):
    
    # move the gradLayer to cuda
    gradLayer.to(device)
    gradSq = gradLayer(img.unsqueeze(0))**2
    
    grad_loss = torch.pow(gradSq.mean(), beta/2)
    
    return grad_loss

def AM():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # undo the above normalization if and when the need arises 
    denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225] )

    Height = 277
    Width = 277
    # generate a numpy array with random values

                # Scharr Filters
    filter_x = np.array([[-3, 0, 3], 
                        [-10, 0, 10],
                        [-3, 0, 3]])
    filter_y = filter_x.T
    grad_filters = np.array([filter_x, filter_y])
    gradLayer = RGBgradients(grad_filters)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Calculations being executed on {}'.format(device))
    model.to(device)
    model.eval()
    for unit_idx in range(32,64):

        img_tensor = torch.rand(1,3,277,277, requires_grad=True, device="cuda")
    #img_tensor = im_tensor.detach().clone().requires_grad_(True).to(device)
        act_wt = 0.5 # factor by which to weigh the activation relative to the regulizer terms
        upscaling_steps = 20 # no. of times to upscale
        upscaling_factor = 1.05
        optim_steps = 60# no. of times to optimize an input image before upscalin
            
        for mag_epoch in range(upscaling_steps+1):
            optimizer = optim.Adam([img_tensor], lr = 0.4)
            
            for opt_epoch in range(optim_steps):
                optimizer.zero_grad()
                d = torch.unsqueeze(img_tensor[:, 2, :, :], dim=1)
                d = torch.cat((d, d, d), dim=1)
                rgb = model.rgb_features(img_tensor[:, :3, :, :])
                d = model.d_features(d)
                x = torch.cat((rgb, d), dim=1)
                layer_out = model.features[:5](x)
                rms = torch.pow((layer_out[0, unit_idx]**2).mean(), 0.5)
                # terminate if rms is nan
                # if torch.isnan(rms):
                #     print('Error: rms was Nan; Terminating ...')
                #     sys.exit()
                
                # pixel intensity
                pxl_inty = torch.pow((img_tensor**2).mean(), 0.5)
                # terminate if pxl_inty is nan
                if torch.isnan(pxl_inty):
                    print('Error: Pixel Intensity was Nan; Terminating ...')
                    sys.exit()
                
                # image gradients
                im_grd = grad_loss(img_tensor[0, :, :, :], beta = 1, device = device)
                # terminate is im_grd is nan
                if torch.isnan(im_grd):
                    print('Error: image gradients were Nan; Terminating ...')
                    sys.exit()
                
                loss = -act_wt*rms + pxl_inty + im_grd        
                # print activation at the beginning of each mag_epoch
                if opt_epoch == 0:
                    print('begin mag_epoch {}, activation: {}'.format(mag_epoch, rms))
                loss.backward()
                optimizer.step()
            # view the result of optimising the image
            print('end mag_epoch: {}, activation: {}'.format(mag_epoch, rms))
            img = image_converter(img_tensor[0, :, :, :])    
            plt.imshow(img)
            plt.title('image at the end of mag_epoch: {}'.format(mag_epoch))
            plt.savefig("features/features.4/%s.png" % unit_idx)
            img = cv2.resize(img, dsize = (0,0), 
                            fx = upscaling_factor, fy = upscaling_factor).transpose(2,0,1) # scale up and move the batch axis to be the first
            img_tensor = normalize(torch.from_numpy(img))[None, :,:,:].to(device).requires_grad_(True)
            
def vis_kernels():
    kernels = model.rgb_features[0].weight.detach().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    img = make_grid(kernels)
    plt.imsave("test.png",img.permute(1, 2, 0).cpu().numpy().astype(np.float32))
    
vis_kernels()
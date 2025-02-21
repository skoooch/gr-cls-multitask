"""
This code visualizes the saliency map of a CNN using integrated gradients,
which highlights which areas in an image influences the model prediction
the most. 

For classification models, we take the class label as the ground
truth for calculating gradients. For grasping models, we take the 
'cos' output of the final layer as the ground truth. Other outputs
such as the 'width', 'sin', etc. could also be used if required.

The saliency maps are generated from the Jacquard Dataset.
The images are saved in the <datasets> folder.

This file has referenced codes from @author: Utku Ozbulak - github.com/utkuozbulak.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""
import matplotlib.pyplot as plt
import os
import cv2
from skimage import transform
from skimage.io import imread, imshow

import numpy as np
import torch
import h5py
from parameters import Params
from PIL import Image
from sklearn.model_selection import train_test_split
from model import Multi_AlexnetMap_Width
from gb_models import WidthFeatures, WidthRgbdFeatures
from guided_backprop import GuidedBackprop

from utils import tensor2img, get_layer_width
from misc_functions import (convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency,
                            save_image,
                            format_np_output)

# Params class containing parameters for AM visualization.
torch.manual_seed(0)
params = Params()
N_IMG = 5
model_type = "grasp"
num_epochs = 10
model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
weights_path = os.path.join(weights_dir, f'alexnetMap_{model_type}.pth')
model =  Multi_AlexnetMap_Width(weights_path, True).to('cuda')
model_weights_path = f"goodale/trained_models/model_{model_type}_move_final.pth"
model.load_state_dict(torch.load(model_weights_path))
model.eval()


# Select data for visualization
bp_data = []
# DataLoader
batch_size = 1
data = torch.load("goodale/rectangle_dataset/indices.pt", weights_only=True)[:, 0, :]
perm = torch.randperm(data.shape[0])
idx = perm[:15000]
data = data[idx]
# data = data[data[:, 3] == 0]
# data = data[data[:, 2] == 0] 
y = (data[:, 0] + 30).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
output_dir = '/scratch/expires-2025-Feb-24/'
os.makedirs(output_dir, exist_ok=True)
filename = f'rect_data_move_test.hdf5'
filepath = os.path.join(output_dir, filename)
h5_file = h5py.File(filepath, 'r')
image_data = h5_file.get("data")
for width in range(30, 120,5):
    for height in range(30, 120,10):
        bp_data.append((width - 30, ((height - 30)//2) , 0, 0))
transformed_imgs = []
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    # Create sub-directory for chosen layer
    save_subdir = "save_subdir_%s" % model_type
    
    # Create submodel with output = selected kernel
    #ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
    ext_model = WidthFeatures(model, vis_layer)

    # Load Jacquard Dataset images
    print('Visualizing saliency maps')
    for id, (inputs) in enumerate(bp_data):
        mean_guided_grads = np.zeros((1, 224, 224))
        shift_x = image_data[inputs[0], inputs[1], 1, 1, 0, 0, 0]
        shift_y = image_data[inputs[0], inputs[1], 1, 1, 0, 0, 1]
        
        for i, kernel_idx in enumerate(range(100)):
            # inputs = img_rgbd
            
            # shape = (batch_size, 4, 224,224, 4)
                 
            # Make predictions for this batch
            img = torch.Tensor(image_data[inputs[0], inputs[1], 0, 0]).to("cuda")
            #shape = (batch_size, 224,224,4)
            img = torch.where(img == 2, 0.1, img)
            img = torch.where(img == 1, -0.1, img)
            prep_img = img.permute(2, 0, 1)[None, :, :, :]
            img_rgbd = img.permute(2,0,1)
            GBP = GuidedBackprop(ext_model)
            # Get gradients
            guided_grads = GBP.generate_gradients(prep_img, kernel_idx)
            guided_grads = guided_grads[:3]
            # Convert to grayscale
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            
            # Mask saliency
            # grayscale_guided_grads = grayscale_guided_grads * np.array(img_mask)

            # Find value of gradient of rank 20th
            flattened_gradient = np.reshape(grayscale_guided_grads, (grayscale_guided_grads.shape[0], -1))
            top_k = int(len(flattened_gradient[0]) * 0.01)
            top_k_grad = np.sort(flattened_gradient, 1)[0][-top_k]
            # Subset gradient to only top 20
            grayscale_guided_grads = np.where(grayscale_guided_grads >= top_k_grad, grayscale_guided_grads, 0)

            mean_guided_grads = mean_guided_grads + grayscale_guided_grads
        width, height, _, _ = inputs
        src = np.array([112- ((height*2) + 30)//2 +shift_y, 112 - (width + 30)//2+shift_x,
            112 + ((height*2) + 30)//2+shift_y, 112 + (width + 30)//2+shift_x,
            112- ((height*2) + 30)//2+shift_y, 112 + (width + 30)//2+shift_x,
            112 + ((height*2) + 30)//2+shift_y, 112 - (width + 30)//2+shift_x,
        ]).reshape((4, 2)) 
     
        dst = np.array([112-45, 112 - 45,
            112 + 45, 112 + 45,
            112- 45, 112 + 45,
            112 + 45, 112 - 45,
        ]).reshape((4, 2))
  
        tform = transform.estimate_transform('projective', src, dst)
        grayscale_guided_grads = mean_guided_grads / 5

        save_img_rgb, save_img_d = img_rgbd[:3, :, :], img_rgbd[3,:,:]

        # Normalize and resize gradient map
        gradient = grayscale_guided_grads - grayscale_guided_grads.min()
        
        gradient /= gradient.max()
        gradient = gradient.transpose(1, 2, 0)
        gradient = cv2.resize(gradient, (save_img_rgb.shape[1], save_img_rgb.shape[2]), cv2.INTER_LINEAR)
        #gradient = gradient * np.array(img_mask)
        
        gradient = np.expand_dims(gradient, 2)
        # Save grayscale gradients
        tf_img = transform.warp(gradient, tform.inverse).transpose(2, 0, 1)
        transformed_imgs.append(tf_img)
        # print(tf_img.sum())
        # print(gradient.sum())

        # save_gradient_images(tf_img, save_subdir,
        #                         'image_%s_layer_%s' % (id, vis_layer) + '_poopy' )
        # save_gradient_images(gradient.transpose(2, 0, 1), save_subdir,
        #                         'image_%s_layer_%s' % (id, vis_layer) + '_Guided_BP_gray' )
        # # Define highlighting color
        # sub_color = np.full((gradient.shape[0], gradient.shape[1], 1), 0.2)
        # main_color = np.full((gradient.shape[0], gradient.shape[1], 1), 0.95)
        # color1 = np.concatenate((main_color, sub_color, sub_color), 2)
        # # Highlight image based on gradient
        # colored_img = save_img_rgb.permute(1, 2, 0).detach().cpu() * 0.5 + gradient * color1
        # colored_img = np.clip(colored_img, 0, 255)
        
        # # Save image
        # path_to_file = os.path.join(save_subdir, 'image_%s_layer_%s' % (id, vis_layer) + '_Guided_BP_img.png')
        # #path_to_file = os.path.join(paths.save_subdir, 'image_%s_layer_%s_kernel_%s_rank_%s' % (id, vis_layer, kernel_idx, i) + '_Guided_BP_img.png')

        # im = Image.fromarray(np.ascontiguousarray(colored_img, dtype=np.uint8))
        # #im = im.resize((256, 256), Image.ANTIALIAS)
        # im.save(path_to_file)

        # #if '%s_image.png' % img_id not in os.listdir(os.path.join(paths.vis_path, paths.main_path)):
        # #    cv2.imwrite(os.path.join(paths.vis_path, paths.main_path, '%s_image.png' % img_id), save_img_rgb)

final_img = np.mean(np.array(transformed_imgs), axis=0, dtype = np.float64)

save_gradient_images(final_img, save_subdir,
                                'image_layer_' + '_final_moved' )
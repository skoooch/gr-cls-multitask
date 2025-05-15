import numpy as np
from sympy import false
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
import sys
import cv2
import os
from utils.utils import get_correct_cls_preds_from_map
from PIL import Image
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v2 import DataLoader
from utils.parameters import Params
import numpy as np
params = Params()

def get_model(model_path, device=params.DEVICE):
    model = Multi_AlexnetMap_v3().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def crop_center(image_array):
    """
    Crop the image from the center to the specified target size.
    
    Args:
        image_array: 3D NumPy array representing the image (height, width, channels).
        target_size: Tuple representing the desired size (height, width).
    
    Returns:
        cropped_array: Cropped NumPy array of size (target_height, target_width, channels).
    """
    current_height, current_width = image_array.shape[:2]
    min_dim = min(current_height, current_width)
    # Calculate the coordinates for the center crop
    start_x = (current_width - min_dim) // 2
    start_y = (current_height - min_dim) // 2
    end_x = start_x + current_width
    end_y = start_y + current_width
    
    # Crop the image
    cropped_array = image_array[start_y:end_y, start_x:end_x]
    res = cv2.resize(cropped_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return res

def rgb_to_depth(rgb_image):
    """
    Convert a 3-channel RGB image where depth is encoded as:
    - Red = Far (higher values mean farther)
    - Blue = Close (higher values mean closer)
    to a single-channel depth map.
    
    Args:
        rgb_image: 3D NumPy array of shape (height, width, 3)
    
    Returns:
        depth_map: 2D NumPy array of shape (height, width), representing depth.
    """
    # Extract the red and blue channels
    red_channel = rgb_image[:, :, 0]
    blue_channel = rgb_image[:, :, 2]
    
    # Normalize both channels to the range [0, 1]
    red_norm = red_channel / 255.0
    blue_norm = blue_channel / 255.0
    
    # Calculate the depth map. Since red = far and blue = close,
    # we might assume that the depth can be represented as red - blue.
    # A higher value in red would mean farther, while a higher value in blue would mean closer.
    depth_map = red_norm - blue_norm
    
    # Normalize the result back to the range [0, 255] for visualization purposes, if needed
    depth_map_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())) * 255
    
    return depth_map_normalized.astype(np.uint8)

def load_images_to_arrays(depth=True):
    image_arrays= []
    depth_path = 'new_data/cleaned'
    rgb_path = 'new_data/RGB'
    # Get a list of all PNG files in the folder
    image_files = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])
    for image_file in image_files:
        # Construct the full file path
        depth_file = os.path.join(depth_path, image_file)
        rgb_file = os.path.join(rgb_path, image_file.replace("Depth", "Color"))
        # Open the image using Pillow
        depth_array = None
        with Image.open(depth_file) as img:
            # Convert the image to a NumPy array and append to the list
            depth_array = crop_center(np.array(img))
            if not depth:
                depth_array = np.zeros(depth_array.shape)
            depth_array = torch.from_numpy(rgb_to_depth(depth_array))[None, None, ...]
        with Image.open(rgb_file) as img:
            # Convert the image to a NumPy array and append to the list
            rgb_array = torch.from_numpy(np.transpose(crop_center(np.array(img))))[None, ...] 
            image_arrays.append((torch.cat((rgb_array, depth_array), dim=1)).to(torch.float).to("cuda") / 255)
    return image_arrays

def get_accuracy(model, images):
    label_order = ["C", "D", "A", "E", "B"]
    labels = ['A', 'B', 'C', 'D', 'E']
    loss = 0
    correct = 0
    total = 0
    labels_repeated = np.repeat(labels, 5)
    for i, (img, label) in enumerate(zip(images, labels_repeated)):
        
        img_test = torch.zeros(img.shape, dtype=torch.float).to("cuda")
        img_test[0, :3, :, :] = img[0, :3, :, :]
        # plt.imshow(img_depth)
        # plt.title("shit")
        # plt.savefig("shit.png")
        # exit() tensor([[ 3611.7319, -3644.3909, -3648.8831, -3639.3789, -3656.8604]],
        label_cls = torch.ones(6, dtype=torch.float32).to("cuda") * -1
        label_cls[label_order.index(label)] = 1.0
        label_cls[5] = 1.0
        output = model(img_test, is_grasp=False)
        batch_correct, batch_total = get_correct_cls_preds_from_map(output,torch.argmax(label_cls))
        correct += batch_correct
        total += batch_total
    print(correct/total)


# images = load_images_to_arrays()
# DEVICE = sys.argv[1]
# MODEL_NAME = params.MODEL_NAME
# MODEL_PATH = params.MODEL_WEIGHT_PATH
# model = get_model(MODEL_PATH, DEVICE)
# get_accuracy(model, images)
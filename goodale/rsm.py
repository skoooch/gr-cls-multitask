from multiprocessing import active_children
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
import sys
import cv2
import os
from PIL import Image
from scipy.spatial.distance import squareform, pdist
from sklearn import manifold, datasets
from model import Multi_AlexnetMap_Width
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3

from parameters import Params
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

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

def get_feature_activations(model, images, labels, layer_i=0):
    activations = {}
    labels_repeated = np.repeat(labels, 5)
    for i, (img, label) in enumerate(zip(images, labels_repeated)):
        if label not in activations.keys(): activations[label] = []
        rgb = img[:, :3, :, :]
        d = torch.unsqueeze(img[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)
        rgb = model.rgb_features(rgb)
        d = model.d_features(d)
        x = torch.cat((rgb, d), dim=1)
        next = model.features[:layer_i](x)
        activations[label].append(next)
    activations_flat = []
    for label in labels:
        for act in activations[label]:
            activations_flat.append(torch.flatten(act).cpu().detach().numpy())
        print(activations_flat[-1].shape)
    act_array = np.asarray(activations_flat)
    return act_array

def get_rgb_activations(model, images, labels, depth=False):
    activations = {}
    labels_repeated = np.repeat(labels, 5)
    for i, (img, label) in enumerate(zip(images, labels_repeated)):
        if label not in activations.keys(): activations[label] = []
        if depth:
            d = torch.unsqueeze(img[:, 3, :, :], dim=1)
            d = torch.cat((d, d, d), dim=1)
            activation = torch.concat((model.rgb_features[0](img[:, :3, :, :]), model.d_features(d)), dim=1)
            activations[label].append(activation)
        else:
            activations[label].append(model.rgb_features[0](img[:, :3, :, :]))
    activations_flat = []
    for label in labels:
        for act in activations[label]:
            activations_flat.append(np.asarray(torch.flatten(act).cpu()))
    act_array = np.asarray(activations_flat)
    return act_array
    
class MDS:
    """ Classical multidimensional scaling (MDS)
                                                                                               
    Args:                                                                               
        D (np.ndarray): Symmetric distance matrix (n, n).          
        p (int): Number of desired dimensions (1<p<=n).
                                                                                               
    Returns:                                                                                 
        Y (np.ndarray): Configuration matrix (n, p). Each column represents a 
            dimension. Only the p dimensions corresponding to positive 
            eigenvalues of B are returned. Note that each dimension is 
            only determined up to an overall sign, corresponding to a 
            reflection.
        e (np.ndarray): Eigenvalues of B (p, ).                                                                     
                                                                                               
    """    
    def cmdscale(D, p = None):
        # Number of points                                                                        
        n = len(D)
        # Centering matrix                                                                        
        H = np.eye(n) - np.ones((n, n))/n
        # YY^T                                                                                    
        B = -H.dot(D**2).dot(H)/2
        # Diagonalize                                                                             
        evals, evecs = np.linalg.eigh(B)
        # Sort by eigenvalue in descending order                                                  
        idx   = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]
        # Compute the coordinates using positive-eigenvalued components only                      
        w, = np.where(evals > 0)
        L  = np.diag(np.sqrt(evals[w]))
        V  = evecs[:,w]
        Y  = V.dot(L)   
        if p and Y.shape[1] >= p:
            return Y[:, :p], evals[:p]
        return Y, evals
    def two_mds(D,p=None):
        my_scaler = manifold.MDS(n_jobs=-1, n_components=2)
        return my_scaler.fit_transform(D)
    def three_mds(D,p=None):
        my_scaler = manifold.MDS(n_jobs=-1, n_components=3)
        return my_scaler.fit_transform(D)
    
params = Params()
activation = {}

def get_model(model_path, device=params.DEVICE):
    model = Multi_AlexnetMap_v3().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

images = load_images_to_arrays(depth=False)
params = Params()
model_type = "grasp"
num_epochs = 10
model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
weights_path = os.path.join(weights_dir, f'alexnetMap_{model_type}.pth')
model =  Multi_AlexnetMap_Width(weights_path, True).to('cuda')


labels = ['A', 'B', 'C', 'D', 'E']
labels_repeated = np.repeat(labels, 5)
# act_array = get_feature_activations(model, images, labels, layer_i=1)
# for i in [1, 5, 8, 11]:
#     act_array = get_feature_activations(model, images, labels, layer_i=i)
#     result = squareform(pdist(act_array, metric="correlation"))
#     np.save("saved_model_rsms/class/features_%s.npy" % (i-1), result)

act_array = get_rgb_activations(model, images, labels)
result = squareform(pdist(act_array, metric="correlation"))
np.save("saved_model_rsms/grasp/rgb.npy", result)

num_images_per_label = 5
# embedding = MDS.cmdscale(result, 2)[0]
# embedding = {cat:embedding[i*num_images_per_label:(i+1)*num_images_per_label] # split into categories
#             for i, cat in enumerate(labels)}   
# ax = plt.gca()
# ax.set_xticks([])
# ax.set_yticks([])
# for cat in labels:
#     ax.scatter(embedding[cat][:, 0],
#                 embedding[cat][:, 1],
#                 label = cat)
# ax.legend()
# plt.savefig('vis/rsm/rgb_1.png')    
# plt.clf()
# ax = plt.gca()
# ax.set_xticks([])
# ax.set_yticks([])
# for cat in labels:
#     avr_x = np.mean(embedding[cat][:, 0])
#     avr_y = np.mean(embedding[cat][:, 1])
#     ax.scatter(avr_x,
#                 avr_y,
#                 label = cat)
# ax.legend()
# plt.savefig('vis/rsm/rgb_1_avr.png')  
mapping = {"A": "figurine", "B": "pen", "C": "chair", "D":"lamp", "E": "plant"}  
embedding = MDS.two_mds(result)
embedding = {cat:embedding[i*num_images_per_label:(i+1)*num_images_per_label] # split into categories
            for i, cat in enumerate(labels)}   

fig = plt.figure()
ax = fig.add_subplot()
for cat in labels:
    ax.scatter(embedding[cat][:, 0],
                embedding[cat][:, 1],
                label=mapping[cat])
    
ax.legend()
plt.title("Layer 1 of RGB_Features (correlation)")
plt.savefig('vis/rsm/rgb_2d_0_correlation_labeled.png')   
plt.clf()

fig = plt.figure()
ax = fig.add_subplot()
for cat in labels:
    avr_x = np.mean(embedding[cat][:, 0])
    avr_y = np.mean(embedding[cat][:, 1])
    ax.scatter(avr_x,
                avr_y,
                label=mapping[cat])
ax.legend()
plt.title("Layer 1 of RGB_Features (AVR) (correlation)")
plt.savefig('vis/rsm/rgb_2d_0_avr_correlation.png')
plt.clf()
embedding = MDS.three_mds(result)
embedding = {cat:embedding[i*num_images_per_label:(i+1)*num_images_per_label] # split into categories
            for i, cat in enumerate(labels)}  

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for cat in labels:
    ax.scatter(embedding[cat][:, 0],
                embedding[cat][:, 1],
                embedding[cat][:, 2],
                label=mapping[cat])   
# for cat in labels:
#     for i in range(len(embedding[cat][:, 0])):
#         ax.text(embedding[cat][i, 0],
#                     embedding[cat][i, 1],
#                     i)
ax.legend()
plt.title("Layer 1 of RGB_Features (correlation)")
plt.savefig('vis/rsm/rgb_3d_0_correlation_labeled.png')   
plt.clf()
mapping = {"A": "figurine", "B": "pen", "C": "chair", "D":"lamp", "E": "plant"}
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for cat in labels:
    avr_x = np.mean(embedding[cat][:, 0])
    avr_y = np.mean(embedding[cat][:, 1])
    avr_z = np.mean(embedding[cat][:, 2])
    ax.scatter(avr_x,
                avr_y,
                avr_z,
                label=mapping[cat])
ax.legend()
plt.title("Layer 1 of RGB_Features (AVR) (correlation)")
plt.savefig('vis/rsm/rgb_3d_0_avr_correlation.png')
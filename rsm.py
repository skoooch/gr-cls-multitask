from multiprocessing import active_children
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
import sys
import os
from PIL import Image
from scipy.spatial.distance import squareform, pdist
from sklearn import manifold, datasets
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v2 import DataLoader
from utils.parameters import Params
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from own_images import load_images_to_arrays

"""
This script calculates representational similarity matrices (RSMs) for a multi-task model.
It loads images, extracts features from the model, computes pairwise distances between feature activations,
and visualizes the results using multidimensional scaling (MDS).
"""
def get_feature_activations(model, images, labels, layer_i=0, top = None,j=1, top_size=13, is_grasp=0):
    """
    Get feature activations from the model.
    Args:
        model: The model to extract activations from.
        images: The images to process.
        labels: The labels for the images.
        layer_i: The layer index to extract features from.
        top: The top indices for the activations.
        j: The index for the activations.
        top_size: The percentage if.
        is_grasp: Whether the task is a grasping task.
    Returns:
        act_array: The activations as a numpy array.
    """
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
        if j == 1:
            activations[label].append(next[:,top[j, :top_size//2,is_grasp],:,:])
        else:
            activations[label].append(next[:,top[j, :,is_grasp],:,:])
    activations_flat = []
    for label in labels:
        for act in activations[label]:
            activations_flat.append(torch.flatten(act).cpu().detach().numpy())
    act_array = np.asarray(activations_flat)
    return act_array

def get_rgb_activations(model, images, labels, depth=False, top = None, is_grasp=0):
    """
    Get RGB activations from the model.
    Args:
        model: The model to extract activations from.
        images: The images to process.
        labels: The labels for the images.
        depth: Whether to include depth information.
        top: The top indices for the activations.
        is_grasp: Whether the task is a grasping task.
    Returns:
        act_array: The activations as a numpy array.
    """
    activations = {}
    labels_repeated = np.repeat(labels, 5)
    for i, (img, label) in enumerate(zip(images, labels_repeated)):
        if label not in activations.keys(): activations[label] = []
        if depth:
            d = torch.unsqueeze(img[:, 3, :, :], dim=1)
            d = torch.cat((d, d, d), dim=1)
            activation = torch.concat((model.rgb_features[0](img[:, :3, :, :])[:,top[0, :,is_grasp],:,:], model.d_features(d)), dim=1)
            activations[label].append(activation)
        else:
            activations[label].append(model.rgb_features[0](img[:, :3, :, :])[:,top[0, :,is_grasp],:,:])
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
DEVICE = sys.argv[1]
MODEL_NAME = params.MODEL_NAME
MODEL_PATH = params.MODEL_WEIGHT_PATH
model = get_model(MODEL_PATH, DEVICE)

data_loader = DataLoader(params.TEST_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
labels = ['A', 'B', 'C', 'D', 'E']
labels_repeated = np.repeat(labels, 5)
selected_kernels = torch.tensor(np.load("shap_arrays/smallest20.npy"), dtype=int).to("cuda")
rsm_folder = "smallest20"

for is_grasp in range(0, 2):
    act_array = get_rgb_activations(model, images, labels, top=selected_kernels, is_grasp=is_grasp)
    result = squareform(pdist(act_array, metric="correlation"))
    if is_grasp == 0:
        np.save(f"saved_model_rsms/class/{rsm_folder}/rgb.npy", result)
    else:
        np.save(f"saved_model_rsms/grasp/{rsm_folder}/rgb.npy", result)
j = 0
for i in [1, 5, 8, 11]:
    j += 1
    for is_grasp in range(0, 2):
        act_array = get_feature_activations(model, images, labels, layer_i=i,top=selected_kernels, j=j, top_size=selected_kernels.shape[1], is_grasp=is_grasp)
        result = squareform(pdist(act_array, metric="correlation"))
        if is_grasp == 0:
            np.save(f"saved_model_rsms/class/{rsm_folder}/features_%s.npy" % (i-1), result)
        else:
            np.save(f"saved_model_rsms/grasp/{rsm_folder}/features_%s.npy" % (i-1), result)
            
exit() # remove this if you want to do the visualization




num_images_per_label = 5

#-------------- From here down is just for visualization ----------------
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
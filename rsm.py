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
from skimage.exposure import match_histograms
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
            if not top == None:
                activations[label].append(next[:,top[j, :top_size//2,is_grasp],:,:])
            else:
                activations[label].append(next[:,:,:,:])
        else:
            if not top == None:
                activations[label].append(next[:,top[j, :,is_grasp],:,:])
            else:
                activations[label].append(next[:,:,:,:])
    activations_flat = []
    for label in labels:
        for act in activations[label]:
            activations_flat.append(torch.flatten(act).cpu().detach().numpy())
    act_array = np.asarray(activations_flat)
    return act_array
def get_head_activations(model, images, labels, layer_i=0, is_grasp=0):
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
        x = model.features(x)
        if is_grasp:
            next = model.grasp[:layer_i](x)
        else:
            next = model.cls[:layer_i](x)
        activations[label].append(next[:,:,:,:])
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
            d = torch.from_numpy(match_histograms(d.cpu().numpy(), np.load("test_depth.npy"))).to("cuda")
            d = torch.cat((d, d, d), dim=1)
            if top:
                activation = torch.concat((model.rgb_features[0](img[:, :3, :, :])[:,top[0, :,is_grasp],:,:], model.d_features[0](d)), dim=1)
            else:
                activation = torch.concat((model.rgb_features[0](img[:, :3, :, :])[:,:,:,:], model.d_features[0](d)), dim=1)
            activations[label].append(activation)
        else:
            if not top == None:
                activations[label].append(model.rgb_features[0](img[:, :3, :, :])[:,top[0, :6,is_grasp],:,:])
            else:
                print(model.rgb_features[0](img[:, :3, :, :]).shape)
                activations[label].append(model.rgb_features[0](img[:, :3, :, :])[:,:,:,:])
    activations_flat = []
    for label in labels:
        for act in activations[label]:
            activations_flat.append(np.asarray(torch.flatten(act).cpu()))
    act_array = np.asarray(activations_flat)
    return act_array
def visualize_rsm(rsm, suffix = "", title = f"Model Activation Representational Dissimilarity Matrix", is_grasp=True):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(rsm, cmap='bwr', aspect='auto', vmin=0, vmax=1)
    num_classes = len(np.unique(labels))
    mapping = {"A": "Figurine", "B": "Pen", "C": "Chair", "D":"Lamp", "E": "Plant"}
    label_order = [mapping[c] for c in ["A","B","C","D","E"]]
    num_images_per_label = rsm.shape[0] // num_classes
    # Draw lines to separate classes
    for i in range(1, num_classes):
        plt.axhline(i * num_images_per_label - 0.5, color='white', linewidth=1)
        plt.axvline(i * num_images_per_label - 0.5, color='white', linewidth=1)
    plt.xticks(
        [i * num_images_per_label + num_images_per_label // 2 for i in range(num_classes)],
        label_order
    )
    plt.yticks(
        [i * num_images_per_label + num_images_per_label // 2 for i in range(num_classes)],
        label_order
    )
    plt.xlabel("Images")
    plt.ylabel("Images")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Dissimilarity')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"rsm_first.png")
        
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

images = load_images_to_arrays(depth=True)
DEVICE = sys.argv[1]
MODEL_NAME = params.MODEL_NAME
MODEL_PATH = params.MODEL_WEIGHT_PATH
model = get_model(MODEL_PATH, DEVICE)

data_loader = DataLoader(params.TEST_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
labels = ['A', 'B', 'C', 'D', 'E']
labels_repeated = np.repeat(labels, 5)
is_grasp = True
#rgb = get_rgb_activations(model, images, labels, depth=True)
# j=0
# for i in [1, 5, 8, 11]:
#     j += 1
#     act_array = get_feature_activations(model, images, labels, layer_i=i, j=j)
#     rgb = np.concatenate((rgb, act_array), axis = 1)
# for i in [1, 3,5, 7]:
#     act_array = get_head_activations(model, images, labels, layer_i=i, is_grasp=is_grasp)
#     rgb = np.concatenate((rgb, act_array), axis = 1)
# print(rgb.shape)
# print(np.load("saved_model_rsms/rgb.npy").shape)
# visualize_rsm(squareform(pdist(rgb, metric="correlation")), title=f"Second Convolutional Layer RDM", is_grasp = is_grasp)
# exit()
is_grasp = 0
rsm_folder = "sort_shap_indices"
selected_kernels = torch.tensor(np.load(f"shap_arrays/{rsm_folder}.npy"), dtype=int).to("cuda")

act_array = get_rgb_activations(model, images, labels,depth=False,top=selected_kernels, is_grasp=is_grasp)
result = squareform(pdist(act_array, metric="correlation"))
print(result.mean())
np.save(f"saved_model_rsms/{"grasp" if is_grasp else "class"}/{rsm_folder}/0.npy", result)

j = 0
for i in [1, 5, 8, 11]:
    j += 1
    act_array = get_feature_activations(model, images, labels, layer_i=i, j=j,top=selected_kernels, top_size=6,is_grasp=is_grasp)
    result = squareform(pdist(act_array, metric="correlation"))
    np.save(f"saved_model_rsms/{"grasp" if is_grasp else "class"}/{rsm_folder}/{j}.npy", result)
    
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
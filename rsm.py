import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
import sys
from scipy.spatial.distance import squareform, pdist

from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v2 import DataLoader
from utils.parameters import Params
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

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



LAYER = 'rgb_features.0'
DEVICE = sys.argv[1]
MODEL_NAME = params.MODEL_NAME
MODEL_PATH = params.MODEL_WEIGHT_PATH
model = get_model(MODEL_PATH, DEVICE)

model.rgb_features[0].register_forward_hook(get_activation(LAYER))
activations = [[],[],[],[],[]]
data_loader = DataLoader(params.TEST_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
labels = data_loader.get_cls_id()
for i, (img, cls_map, label) in enumerate(data_loader.load_cls()):
    model(img, is_grasp=False)
    activations[label.item()].append(activation[LAYER])
activations_flat = []
for i in range(5):
    for act in activations:
        activations_flat.append(act.cpu())
result = squareform(pdist(activations_flat, metric = 'correlation'))

num_images_per_label = len(activations[0])
embedding = MDS.cmdscale(result, 2)[0]
embedding = {cat:embedding[i*num_images_per_label:(i+1)*num_images_per_label] # split into categories
            for i, cat in enumerate(DataLoader.get_cls_id())}   
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
for cat in labels:
    ax.scatter(embedding[cat][:, 0],
                embedding[cat][:, 1],
                label = cat)
plt.savefig('vis/rsm/rgb_0_cls.png')    

import copy
import os
import sys

import h5py
import numpy as np
from sympy import acot
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchsummary import summary
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v2 import DataLoader
from training.single_task.evaluation import get_cls_acc, get_grasp_acc
from utils.parameters import Params
LAYERS = ['first','features.0','features.4', 'features.7', 'features.10']
SIZES = [128,32,64,64,64]
params = Params()
layers = []
def remove_connections(model: nn.Module, layer: str, removed_connections: list) -> nn.Module:
    """
    Silence the impact of specific connections (src_kernel, tgt_kernel) in <layer> of <model>.
    """
    with torch.no_grad():
        for name, W in model.named_parameters():
            if name == layer+'.weight':
                for (src, tgt) in removed_connections:
                    # Set the connection from src (in_channel) to tgt (out_channel) to zero
                    W.data[tgt, src, :, :] = 0
    return model

def remove_players(model: nn.Module, layer: str, removed_idx: list) -> nn.Module:
    """
    Silence the impact of weights within the <layer> of the <model>, except those in <weights>.
    Silenced weights are replaced by the mean weights of other functional weights
    """
    # Update new weights onto new model
    with torch.no_grad():
        for name, W in model.named_parameters():
            if name == layer+'.weight' or name == layer+'.bias':           
                # Calculate mean non-removed weight
                keeping_idx = [i for i in range(W.data.shape[0]) if i not in removed_idx]
                w_mean = torch.mean(W.data[keeping_idx], dim=0)
                W.data[removed_idx] = w_mean                
    return model
        

def one_iteration(
    model, 
    layer,
    players,
    c,
    truncation,
    device='cuda',
    chosen_players=None,
    metric='accuracy', 
    task='grasp',
    activations=None,
    indices_keys =None):
    '''One iteration of Neuron-Shapley algoirhtm.'''
    # Original performance of the model with all players present.
    init_val = [85,81.5][task == 'grasp']
    
    # A random ordering of players
    idxs = np.random.permutation(len(c))
    # -1 default value for players that have already converged
    marginals = -np.ones(len(c))
    marginals[chosen_players] = 0.

    truncation_counter = 0
    old_val = init_val
    
    for i,idx in enumerate(idxs):
        model = remove_connections(model, LAYERS[layer_i], [idx_to_val[idx]])
        
        new_val = get_acc(model, task=task, device=device)
        print(f'{i} Accuracy: {new_val}')
        marginals[idx] = old_val - new_val
        old_val = new_val
        if metric == 'accuracy' and new_val <= truncation:
            truncation_counter += 1
        else:
            truncation_counter = 0
        if truncation_counter > 5:
            break
    
    return idxs.reshape((1, -1)), marginals.reshape((1, -1))


def get_acc(model, shap_mask=[], activations=None, task='cls', device='cuda:0', skip=1):
    if task == 'cls':
        return get_cls_acc(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=params.DATA_TRUNCATION, device=device, shap_mask=shap_mask, activations=activations, skip = skip)[0]
    elif task == 'grasp':
        return get_grasp_acc(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=params.DATA_TRUNCATION, device=device, shap_mask=shap_mask, activations=activations,skip = skip)[0]
    else:
        raise ValueError('Invalid task!')


class TensorID:
    """
    ID wrapper around a tensor
    """
    def __init__(self, layer: str, tensor: torch.Tensor, id_: int):
        self.layer = layer
        self.tensor = tensor
        self.id = id_

    def __str__(self):
        return f"<id:{self.id}, tensor:{self.tensor}>"

    def __repr__(self):
        return self.__str__()


def get_model(model_path, device=params.DEVICE):
    model = Multi_AlexnetMap_v3().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def get_weights(model, layer):
    if layer == "first":
        weights = []
        biases = []
        for layer in ["rgb_features.0", "d_features.0"]:
            for name, W in model.named_parameters():
                if name == layer+'.weight':
                    weight = torch.split(W, 1, dim=0)
                elif name == layer+'.bias':
                    bias = torch.split(W, 1, dim=0)

            assert weight is not None, f"Layer {layer} not found"
            assert bias is not None, f"Layer {layer} not found"

            weights += [TensorID(layer+'.weight', None, i) for i, _ in enumerate(weight)]
            biases += [TensorID(layer+'.bias', None, i) for i, _ in enumerate(bias)]

        return weights, biases
    else:        
        for name, W in model.named_parameters():
            if name == layer+'.weight':
                weight = torch.split(W, 1, dim=0)
            elif name == layer+'.bias':
                bias = torch.split(W, 1, dim=0)

        assert weight is not None, f"Layer {layer} not found"
        assert bias is not None, f"Layer {layer} not found"

        weights = [TensorID(layer+'.weight', None, i) for i, _ in enumerate(weight)]
        biases = [TensorID(layer+'.bias', None, i) for i, _ in enumerate(bias)]

        return weights, biases

def convert_text_array_to_tuple_array(array):
    return_arr = []
    for s in array:
        tup = s.split('_')
        return_arr.append((tup[0], tup[1])) 
    return return_arr
def convert_tuple_array_to_text_array(array):
    return_arr = []
    for s in array:
        return_arr.append(f"{s[0]}_{s[1]}") 
    return return_arr
def convert_tuple_array_to_range(array, top_keys):
    return_arr = []
    for s in array:

        i = np.where(top_keys == int(s[0]))[0]*SIZES[layer_i] + int(s[1])
        return_arr.append(i) 
    return return_arr
def convert_index_to_value(idx, top_keys):
    src = top_keys[idx // SIZES[layer_i]]
    tgt = idx % SIZES[layer_i]
    return (src, tgt)
    
    
def get_players(directory, src_kernels, tgt_kernels):
    ## Load the list of all players (filters) else save
    players = []
    for i in src_kernels:
        for j in tgt_kernels:
            players.append(f'{i}_{j}')
    open(os.path.join(directory, 'players.txt'), 'w').write(','.join(players))
    p = convert_text_array_to_tuple_array(players)
    return p


def instantiate_tmab_logs(players, log_dir):
    ## Create placeholder for results in save ASAP to prevent having the 
    ## same expriment_number with other parallel cb_run.py scripts
    mem_tmc = np.zeros((0, len(players)))
    idxs_tmc = np.zeros((0, len(players))).astype(int)

    with h5py.File(log_dir, 'w') as foo:
        foo.create_dataset("mem_tmc", data=mem_tmc, compression='gzip')
        foo.create_dataset("idxs_tmc", data=idxs_tmc, compression='gzip')

    return mem_tmc, idxs_tmc

def average_activations(model):
    activations = []
    data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
    labels = data_loader.get_cls_id()
    for i, (img, cls_map, label) in enumerate(data_loader.load_cls()):
        if(i%7 == 0):
            rgb = img[:, :3, :, :]
            d = torch.unsqueeze(img[:, 3, :, :], dim=1)
            d = torch.cat((d, d, d), dim=1)
            #NEED TO CHANGE BASED ON LAYER
            rgb = model.rgb_features(rgb)
            d = model.d_features(d)
            x = torch.cat((rgb, d), dim=1)
            activations.append(x[0].cpu().detach().numpy())
            # First layer
            # activations.append(model.rgb_features[0](img[:, :3, :, :])[0].cpu().detach().numpy()[0])
    np_activations_mean = np.zeros(activations[0].shape)
    for activation in activations:
        np_activations_mean += activation
    act_tensor = torch.tensor(np_activations_mean / len(activations))
    torch.save(act_tensor, os.path.join('shap/activations', 'x.pt'))
    
def compute_avg_connection_activations(model, layer_i, data_loader):
    """
    Returns avg_conn_activations: [out_channels, in_channels, H, W]
    """
    # Get layer weights
    layer = dict(model.named_modules())[LAYERS[layer_i]]
    W = layer.weight.data  # [out_channels, in_channels, kH, kW]
    out_channels, in_channels, kH, kW = W.shape

    # Get a batch of input images
    avg_conn_activations = None
    n_samples = 0
    for i, (img, _, _) in enumerate(data_loader.load_cls()):
        if(i%7 == 0):
            rgb = img[:, :3, :, :]
            d = torch.unsqueeze(img[:, 3, :, :], dim=1)
            d = torch.cat((d, d, d), dim=1)
            #NEED TO CHANGE BASED ON LAYER
            rgb = model.rgb_features(rgb)
            d = model.d_features(d)
            x = torch.cat((rgb, d), dim=1)
            if layer_i == 0:
                pass
            else:
                x = model.features[:[0,0,4,7,10][layer_i]](x)
            # For each connection, compute its contribution
            batch_conn_acts = []
            for tgt in range(out_channels):
                tgt_acts = []
                for src in range(in_channels):
                
                    
                    # Only the src channel
                   
                    x_src = x[:, src:src+1, :, :]
                    # Only the tgt kernel for src
                    w = W[tgt:tgt+1, src:src+1, :, :]
                    # Convolve
                    act = torch.nn.functional.conv2d(x_src, w)
                    tgt_acts.append(act.detach().cpu().numpy())
                batch_conn_acts.append(np.stack(tgt_acts, axis=0))  # [in_channels, batch, H, W]
            batch_conn_acts = np.stack(batch_conn_acts, axis=0)  # [out_channels, in_channels, batch, H, W]
            if avg_conn_activations is None:
                avg_conn_activations = batch_conn_acts
            else:
                avg_conn_activations += batch_conn_acts
            n_samples += 1
            
    avg_conn_activations = avg_conn_activations / n_samples  # [out_channels, in_channels, batch, H, W]
    # Average over batch dimension
    avg_conn_activations = np.mean(avg_conn_activations, axis=2)  # [out_channels, in_channels, H, W]
    np.save(f"shap/activations/connections_{layer_i}.npy", avg_conn_activations)
if __name__ == "__main__":
    # Experiment parameters
    SAVE_FREQ = 100

    TASK = 'grasp'
    layer_i = 1
    LAYER = LAYERS[layer_i]
    METRIC = 'accuracy'
    TRUNCATION_ACC = 50.
    DEVICE = sys.argv[1]
    DIR = 'shap/connections/'
    MODEL_NAME = params.MODEL_NAME
    MODEL_PATH = params.MODEL_WEIGHT_PATH

    PARALLEL_INSTANCE = sys.argv[2]

    ## CB directory
    run_name = '%s_%s_%s' % (MODEL_NAME, LAYER, TASK)
    run_dir = os.path.join(DIR, run_name)
    log_dir = os.path.join(run_dir, '%s_%s.h5' % (run_name, PARALLEL_INSTANCE))



    if run_name not in os.listdir(DIR):
        os.mkdir(run_dir)

    ## Load Model and get weights
    model = get_model(MODEL_PATH, DEVICE)

    weights, bias = get_weights(model, LAYER)
    weights = weights#[:-2]
    data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
    shap_values = np.load("shap_arrays/shap_values.npy")
    shap_values[:,:,0] /= 86
    shap_values[:,:,1] /= 81
    # Get indices of top 5 values along the last axis
    shap_values = np.mean(shap_values, axis=-1)

    top5_indices = np.argsort(shap_values, axis=-1)[..., -5:]
    top_5_keys = top5_indices[layer_i,:]
    players = get_players(run_dir, top5_indices[layer_i,:], range(SIZES[layer_i]))

    # Instantiate tmab logs
    mem_tmc, idxs_tmc = instantiate_tmab_logs(players, log_dir)

    ## Running CB-Shapley
    #c = {i: np.array([i]) for i in range(len(players))}

    c = set([player for player in players])
    val_to_idx = {}
    idx_to_val = {}
    for p in range(len(c)):
        val_to_idx[convert_index_to_value(p, top_keys=top_5_keys)] = p
        idx_to_val[p] = convert_index_to_value(p, top_keys=top_5_keys)

    counter = 0
    while True:
        ## Load the list of players (filters) that are determined to be not confident enough
        ## by the cb_aggregate.py running in parallel to this script
        if 'chosen_players.txt' in os.listdir(run_dir):
            chosen_players = open(os.path.join(run_dir, 'chosen_players.txt')).read()
            chosen_players = np.array(chosen_players.split(',')).astype(np.int32)

            if len(chosen_players) == 1:
                break
        else:
            chosen_players = None  
        idxs, vals = one_iteration(
            copy.deepcopy(model),
            LAYER, 
            players,
            c,
            TRUNCATION_ACC,
            device=DEVICE,
            chosen_players=chosen_players,
            metric=METRIC,
            task=TASK,
            indices_keys=top_5_keys
        )

        mem_tmc = np.concatenate([mem_tmc, vals])
        idxs_tmc = np.concatenate([idxs_tmc, idxs])
        
        ## Save results every SAVE_FREQ iterations
        if counter % SAVE_FREQ == SAVE_FREQ - 1:
            with h5py.File(log_dir, 'w') as foo:
                foo.create_dataset("mem_tmc", data=mem_tmc, compression='gzip')
                foo.create_dataset("idxs_tmc", data=idxs_tmc, compression='gzip')
                
        counter += 1
        print(counter)

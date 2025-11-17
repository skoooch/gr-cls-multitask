from math import nan
import math
from cv2 import norm, threshold
import torch
import os
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import font_manager
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from training_utils.evaluation import get_cls_acc, get_grasp_acc, denormalize_grasp, map2singlegrasp
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy import stats

from data_processing.data_loader_v2 import DataLoader
from utils.utils import get_correct_cls_preds_from_map
from utils.grasp_utils import get_correct_grasp_preds
from itertools import groupby
from operator import itemgetter

font_manager._load_fontmanager(try_read_cache=False)
font_path = 'ARIAL.TTF'  # Replace with the actual path
font_entry = font_manager.FontEntry(fname=font_path, name='MyCustomFontName')
font_manager.fontManager.ttflist.insert(0, font_entry) # Add to the beginning of the list
plt.rcParams['font.family'] = ['MyCustomFontName'] # Set as default
params = Params()

data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)
SEED=42


def get_cls_dist(model, dataset=params.TEST_PATH, truncation=None, dissociate=[]):
    correct = 0
    total = 0
    dist = np.zeros(400)
    if(dataset == params.TEST_PATH):
        for i, (img, _, label) in enumerate(data_loader.load_cls()):
            if truncation is not None and (i * params.BATCH_SIZE / data_loader.n_data) > truncation:
                break
            output = model(img, is_grasp=False,dissociate=dissociate)
            batch_correct, batch_total = get_correct_cls_preds_from_map(output, label)
            correct += batch_correct
            total += batch_total
            dist[i] = batch_correct*100.0
    return dist        

def get_grasp_dist(model, dataset=params.TEST_PATH, truncation=None, dissociate=[]):
    """Returns the test accuracy and loss of a Grasp model."""
    correct = 0
    total = 0
    dist = np.zeros(400)
    if(dataset == params.TEST_PATH):
        for i, (img, map, candidates) in enumerate(data_loader.load_grasp()):
            if truncation is not None and (i * params.BATCH_SIZE / data_loader.n_data) > truncation:
                break
            output = model(img, is_grasp=True,dissociate=dissociate)
            # Move grasp channel to the end
            output = torch.moveaxis(output, 1, -1)
            # Denoramlize grasps
            denormalize_grasp(output)

            # Convert grasp map into single grasp prediction
            output_grasp = map2singlegrasp(output)
            output_grasp = torch.unsqueeze(output_grasp, dim=1).repeat(1, candidates.shape[0], 1)
            batch_correct, batch_total =  get_correct_grasp_preds(output_grasp, candidates[None, :, :]) #get_correct_grasp_preds_from_map(output, map)
            correct += batch_correct
            total += batch_total
            dist[i] = batch_correct*100.0
    return dist

if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_name = params.MODEL_NAME
    weights_dir = params.MODEL_PATH
    weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
    # AlexNet with 1st, 2nd layer pretrained on Imagenet
    model =  Multi_AlexnetMap_v3().to(params.DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    path = params.TEST_PATH   
        
    ###-------- Change these values for different layer and anaysis -----------
    diff = True
    layer = int(0 if len(sys.argv) < 2 else sys.argv[1])
    load_data = True
    half = True
    normalize_accuracies = False
    inverse_coloring = False
    ### ------------------------------

    if diff: top = torch.tensor(np.load("shap_arrays/sort_shap_indices_diff_depth.npy"), dtype=int)
    else: top = torch.tensor(np.load("shap_arrays/sort_shap_indices_depth.npy"), dtype=int)
    data_length = [128,32,64,64,64][layer]
    if half: data_length = data_length // 2


    if not load_data:
        data = np.zeros((data_length, 2, 2, 400), np.float16)
        for lesion_i in range(data_length):
            for i in range(2):
                dissociate = top[:,:lesion_i,i]
                dissociate[:layer, :] = 0
                dissociate[layer + 1:, :] = 0
                # Get test acc for CLS model            
                data[lesion_i, i, 0] = get_cls_dist(model, include_depth=True, seed=None, dataset=path, truncation=None, dissociate=dissociate)
                data[lesion_i, i, 1] = get_grasp_dist(model, include_depth=True, seed=None, dataset=path, truncation=None, dissociate=dissociate)
                
        if not diff: np.save(f'dissociation_accuracies/layer_{layer}_data.npy', data)
        else: np.save(f'dissociation_accuracies/diff_layer_{layer}_data.npy', data)
    else:
        if not diff: data = np.load(f'dissociation_accuracies/layer_{layer}_data.npy')
        else: data = np.load(f'dissociation_accuracies/diff_layer_{layer}_data.npy')
        if normalize_accuracies:
            accuracies = np.mean(data, axis=-1)
            # Normalize accuracies for each task separately (across lesion indices)
            normalized_accuracies = np.zeros_like(accuracies)
            for task_idx in range(2):
                task_acc = accuracies[:, :, task_idx]  # shape: (data_length, 2)
                min_acc = np.min(task_acc)
                max_acc = np.max(task_acc)
                # Normalize to range [0.2, 0.85]
                if max_acc - min_acc != 0:
                    normalized_accuracies[:, :, task_idx] = 0.2 + (task_acc - min_acc) * (0.85 - 0.2) / (max_acc - min_acc)
                else:
                    normalized_accuracies[:, :, task_idx] = 0.2
            fake_data = np.zeros_like(data)
            for lesion_i in range(fake_data.shape[0]):
                for i in range(2):
                    for j in range(2):
                        num_correct = int(normalized_accuracies[lesion_i, i, j] * 400)
                        fake_data[lesion_i, i, j, :num_correct] = 1
            data[3,:,]
            data = fake_data
        else:
            data = data/100
    # Plot the data array as two separate plots

    data_lenth = int(data_length)
    x = np.arange(data_length)

    #### T CURVES ###########3


    # ...existing code for data generation...
    accuracies = np.mean(data, axis=-1)
    # Calculate t-values for each curve compared to the baseline (lesion_i=0)
    t_values = np.zeros((data_length, 2,2))
    baseline = [0.85, 0.85] if normalize_accuracies else [.85, .815]

    for lesion_i in range(data_length):
        for i in range(2):  # 0: class kernels, 1: grasp kernels
            for j in range(2):  # 0: classification accuracy, 1: grasp accuracy
                # t-test: compare current accuracy to baseline
                
                t_stat, _ = stats.ttest_1samp(data[lesion_i, i, j], baseline[j], alternative="greater")
                if accuracies[lesion_i, i, j] == 1.0: t_stat = 0
                t_values[lesion_i, i, j] = -t_stat
                if -t_stat == np.inf or t_stat == np.inf: 
                    print(data[lesion_i, i, j])
    t_values[0, :,:] = 0
    sig_tvals = np.zeros((data_length, 2))  # [lesion_i, task]

    for lesion_i in range(data_length):
        for task in range(2):  # 0: classification, 1: grasp
            if inverse_coloring:
                acc_cls = data[lesion_i, task, 0, :]
                acc_grasp = data[lesion_i, task, 1, :]
            else:
                acc_cls = data[lesion_i, 0, task, :]
                acc_grasp = data[lesion_i, 1, task, :]

            t_stat, p_val = stats.ttest_ind(acc_cls, acc_grasp,alternative="less")
            if task == 0: t_stat *= -1 
            sig_tvals[lesion_i, task] = t_stat

    # Plot t-value curves
    plt.figure(figsize=(10, 8))
    x = (x/(len(x)*2)) * 100
    line_width = 3
    np.savetxt(f"dissociation_accuracies/{layer}.txt", t_values.reshape(-1,4),fmt='%1.2f', delimiter=',')
    plt.plot(x, t_values[:, 0, 0], label='T-value: Classification (Classification Kernels)', linestyle='-', color='red', linewidth=line_width)
    plt.plot(x, t_values[:, 0, 1], label='T-value: Grasp (Classification Kernels)', linestyle='--', color='blue' if not inverse_coloring else 'red', linewidth=line_width)
    plt.plot(x, t_values[:, 1, 1], label='T-value: Grasp (Grasp Kernels)', linestyle='-', color='blue', linewidth=line_width)
    plt.plot(x, t_values[:, 1, 0], label='T-value: Classification (Grasp Kernels)', linestyle='--', color='blue' if inverse_coloring else 'red', linewidth=line_width)
    # # Plot a purple horizontal line where anova_f_values > 4.0
    above_threshold = sig_tvals > 2.33 
    above_threshold[0, :] = 0
    added_labels = [False, False]  # [classification, grasping]

    for task_i in range(2):
        if np.any(above_threshold[:, task_i]):
            indices = np.where(above_threshold[:, task_i])[0]
            for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
                group = list(map(itemgetter(1), g))
                
                global_min_y = np.min(t_values)
                global_max_y = np.max(t_values)
                y_line = global_min_y - 0.2 * abs(global_min_y) - (2 * task_i) * global_max_y/100 - 0.5
                label = None
                if not added_labels[task_i]:
                    label = f'Effect of {["Classification", "Grasping"][task_i]} Lesion: T-value > 2.33'
                    added_labels[task_i] = True
                group = (np.array(group)/(data_length*2)) * 100
                plt.hlines(y=y_line, xmin=group[0], xmax=group[-1], colors=['red', 'blue'][task_i], linewidth=4, label=label)

    plt.axhline(y=2.58, color='black', linestyle='dotted', label='Significance threshold (t, p=0.01)')
    # Show all x-ticks, but only label multiples of 10
    xticks = np.arange(0, int(np.ceil(x[-1])) + 1)
    xtick_labels = [str(tick) if tick % 10 == 0 else '' for tick in xticks]
    plt.xticks(xticks, xtick_labels, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Percentage of Feature Maps Lesioned', fontsize=26)
    plt.ylabel('T-value', fontsize=26)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.tight_layout()
    if diff:
        plt.savefig(f'vis/lesion2/{'inverse_coloring/' if inverse_coloring else ''}diff_tvalue_curves_layer_{layer}{'_half' if half else ''}{'_normal' if normalize_accuracies else ''}.png', dpi=300)
    else:
        plt.savefig(f'vis/lesion2/{'inverse_coloring/' if inverse_coloring else ''}tvalue_curves_layer_{layer}{'_half' if half else ''}{'_normal' if normalize_accuracies else ''}.png', dpi=300)
    plt.close()


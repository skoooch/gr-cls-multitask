import enum
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pylab as plt
from matplotlib import font_manager
from tqdm import tqdm
import rsatoolbox
from process_data import get_data_matlab
import sys
import re
import os

from PIL import Image
from scipy.spatial.distance import squareform, pdist
from sklearn import manifold, datasets
import numpy as np
from shapley_analysis import confidence_interval
from scipy import stats
from scipy.stats import pearsonr, spearmanr

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

def perform_rsm_vis(times, time_region=0,task="class"):
    data = get_data_matlab(task,avr=True, left = False)
    labels = data.keys()
    mapping = {"A": "figurine", "B": "pen", "C": "chair", "D":"lamp", "E": "plant"}
    label_order = [mapping[c] for c in ["A","B","C","D","E"]]
    activations_flat = []
    time_period = (np.where(timepoints == times[time_region][0])[0][0], np.where(timepoints == times[time_region][1])[0][0])
    points_per_object = {}
    for cat in label_order:
        points_per_object[cat] = 0
        for object_data in data[cat]:
            relevant_signal = object_data[time_period[0]:time_period[1], :]
            activations_flat.append(relevant_signal.flatten())
            points_per_object[cat] += 1
    act_array = np.asarray(activations_flat)
    
    result = squareform(pdist(act_array, metric="correlation")) #EEG RSM is calculated here!!
    embedding = MDS.two_mds(result) 
    total_objects_sofar = 0
    embedding_categorized = {}
    for cat in labels:
        embedding_categorized[cat] = embedding[total_objects_sofar:total_objects_sofar + points_per_object[cat]]
        total_objects_sofar += points_per_object[cat]
    fig = plt.figure()
    ax = fig.add_subplot()
    for cat in labels:
        ax.scatter(embedding_categorized[cat][:, 0],
                    embedding_categorized[cat][:, 1],
                    label=cat)
    ax.legend()
    plt.title("%sms to %sms %s (correlation)" % (times[time_region][0], times[time_region][1], task))
    plt.savefig('vis/%s/ts%s_correlation.png' % (task,(time_region+1)))   
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    for cat in labels:
        avr_x = np.mean(embedding_categorized[cat][:, 0])
        avr_y = np.mean(embedding_categorized[cat][:, 1])
        ax.scatter(avr_x,
                    avr_y,
                    label=[cat])
    ax.legend()
    plt.title("%sms to %sms Averaged %s (correlation)" % (times[time_region][0], times[time_region][1], task))
    plt.savefig('vis/%s/ts%s_correlation_avr.png' % (task, (time_region+1)))   
    plt.clf()
        
def visualize_rsm(rsm, suffix = "", title = "Recognition task-Related EEG RDM (175–225 ms) from Posterior Brain Regions"):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(rsm, cmap='bwr', aspect='auto', vmin=0, vmax=1)
    num_classes = 5
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
    plt.savefig(f"rsm_{suffix}.png")
def time_eeg_rsm(task):
    time_bin = 2
    data = get_data_matlab(task, avr=True, left=False, bad_participants=([16] if task=="class" else [18,19,20]))
    data = np.array(list(data.values()))
    activations_flat = []
    zero_idx = np.where(timepoints == times[0][0])[0][0]
    end_idx = np.where(timepoints == times[-1][1])[0][0]
    i = zero_idx
    averaged_data = np.mean(data, axis=(0, 1,))
    print(averaged_data.shape)
    print(end_idx)
    while i + time_bin < end_idx:
        activations_flat.append(averaged_data[i:i+time_bin].flatten())
        i += time_bin
    act_array = np.asarray(activations_flat)
    print(act_array.shape)
    rsm = squareform(pdist(act_array, metric="correlation"))  # EEG RSM is calculated here!!
    plt.figure(figsize=(8, 6))
    im = plt.imshow(rsm, cmap='bwr', aspect='auto', vmin=0, vmax=1)
    plt.xlabel("Time from Stimulus Onset (ms)")
    plt.ylabel("Time from Stimulus Onset (ms)")
    # Set x and y axis ticks to a subset of ms given by timepoints (5-10 evenly spaced)
    num_ticks = min(10, act_array.shape[0])
    tick_indices = np.linspace(0, act_array.shape[0] - 1, num=num_ticks, dtype=int)
    ms_ticks = [int(timepoints[zero_idx + i * time_bin]) for i in tick_indices]
    plt.xticks(ticks=tick_indices, labels=ms_ticks)
    plt.yticks(ticks=tick_indices, labels=ms_ticks)  
    plt.gca().invert_yaxis()  # Reverse the y-axis direction
    plt.title(f"RDM for EEG data across time - {'Recognition' if task == 'class' else 'Grasping'} Task - Posterior Electrodes")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Dissimilarity')
    plt.tight_layout()
    plt.savefig(f"vis/temporal/rsm_{suffix}_{task}_full.png")
def visualize_eeg_rsm(task):
    data = get_data_matlab(task,avr=True, left = False)
    mapping = {"A": "figurine", "B": "pen", "C": "chair", "D":"lamp", "E": "plant"}
    label_order = [mapping[c] for c in ["A","B","C","D","E"]]
    activations_flat = []
    time_period = (np.where(timepoints == times[3][0])[0][0], np.where(timepoints == times[3][1])[0][0])
    points_per_object = {}
    for cat in label_order:
        points_per_object[cat] = 0
        for object_data in data[cat]:
            relevant_signal = object_data[time_period[0]:time_period[1], :]
            activations_flat.append(relevant_signal.flatten())
            points_per_object[cat] += 1
    act_array = np.asarray(activations_flat)
    
    result = squareform(pdist(act_array, metric="correlation")) #EEG RSM is calculated here!!
    visualize_rsm(result, f"eeg_single", title = "Recognition task-Related EEG RDM (125–175 ms) from Posterior Brain Regions")
      
def comparative_analysis(model_rsm_path, timepoints, times, task="cls",corr_type="pearson", name_suffix="plot",  single = -1, bad_participants =[]):
    """
    Compare the RSM of EEG data with the RSM of model data.
    Args:
        model_rsm_path (str): Path to the folder containing .npy files.
        timepoints (np.ndarray): Array of time points.
        times (list): List of tuples representing time periods.
        task (str): Task type ("cls" or "grasp").
        name_suffix (str): Suffix for the output file name.
    """
    mapping = {"A": "figurine", "B": "pen", "C": "chair", "D":"lamp", "E": "plant"}
    label_order = [mapping[c] for c in ["A","B","C","D","E"]]
    corrs = []
    data = get_data_matlab(task,avr=True, left=False, single = single, bad_participants=bad_participants)
    if "depth" not in model_rsm_path: model_order = ["rgb.npy", "features_0.npy", "features_4.npy", "features_7.npy", "features_10.npy"]
    else: model_order = ["0.npy", "1.npy","2.npy","3.npy","4.npy"]

    for i in range(len(times)):
        corrs.append([])
        labels = data.keys()
        activations_flat = []
        time_period = (np.where(timepoints == times[i][0])[0][0], np.where(timepoints == times[i][1])[0][0])
        points_per_object = {}
        for cat in label_order:
            points_per_object[cat] = 0
            for object_data in data[cat]:
                relevant_signal = object_data[time_period[0]:time_period[1], :]
                activations_flat.append(relevant_signal.flatten())
                points_per_object[cat] += 1
        act_array = np.asarray(activations_flat)
        
        result = squareform(pdist(act_array, metric="correlation")) #EEG RSM is calculated here!!
        
        # List all .npy files in the folder
        
        npy_files = [f for f in os.listdir(model_rsm_path) if f.endswith('.npy')]
        matrices = {}
        for file in npy_files:
            file_path = os.path.join(model_rsm_path, file)
            matrices[file] = np.load(file_path)  # Load and store each matrix in the dictionary
        for file in model_order:
            model_matrix = matrices[file]
            rsm1_flat = result[np.triu_indices(result.shape[0], k=1)]
            rsm2_flat = model_matrix[np.triu_indices(model_matrix.shape[0], k=1)]
            ## THIS IS WHERE THE CORRELATION between eeg and model IS CALCULATED
            if corr_type=="spearman": corrs[i].append(spearmanr(rsm1_flat, rsm2_flat))
            elif corr_type=="pearson": corrs[i].append(pearsonr(rsm1_flat, rsm2_flat))
    # Plot for each time period
    corrs = np.array(corrs)
    N = len(corrs)
    for i, cor in enumerate(np.array(corrs)):
        # Example correlation data for one layer in one time window
        # Replace this with your actual correlation data # Sample data

        # Conducting a one-sample t-test (one-tailed) against zero
        t_stat, p_value = stats.ttest_1samp(cor[:, 0], 0)

        # Adjusting for one-tailed test (since scipy gives a two-tailed p-value)
        one_tailed_p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        print(f"time period {i+1}, {task}, p-value (1-tailed against 0): {one_tailed_p_value}")
    ind = np.arange(N)  
    width = 0.13 #0.15 width of the bars
    axis_font_size = 18
    title_font_size = 20
    tick_font_size = 13
    plt.figure(figsize=(20, 6))
        # Set the default font family to sans-serif
    matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    font_path = 'ARIAL.TTF'  # Replace with the actual path
    font_entry = font_manager.FontEntry(fname=font_path, name='MyCustomFontName')
    font_manager.fontManager.ttflist.insert(0, font_entry) # Add to the beginning of the list
    plt.rcParams['font.family'] = ['MyCustomFontName'] # Set as default
    bars = []
    n = 325
    for i in range(5):   
        vals = corrs[:, i, 0]
        vals[vals < 0] = 0
        # confidence_intervals = [confidence_interval(r, n) for r in corrs[:, i, 0]]
        # print(confidence_intervals)
        # # Extract lower and upper bounds
        # lower_bounds = [ci[0] for ci in confidence_intervals]
        # upper_bounds = [ci[1] for ci in confidence_intervals]
        if task == "grasp":
            bar = plt.bar(ind + width*i, vals,  width, color = (0.2, 0.2, 1, 1- 0.15*i)) 
        else:
            bar = plt.bar(ind + width*i, vals,  width, color = (1, 0.2, 0.2, 1- 0.15*i))
        bars.append(bar)
        for j, rect in enumerate(bar):
            height = rect.get_height()
            if corrs[j, i][1] < 0.05 and corrs[j,i][0] > 0 and corrs[j, i][1] > 0.01:
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, "*", ha='center', va='bottom')
            elif corrs[j,i][0] > 0 and corrs[j, i][1] < 0.01:
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, "**", ha='center', va='bottom')
    ticks = [f'{desire_times[i][0]}-{desire_times[i][1]}' for i in range(len(desire_times))]
    plt.xticks(ind+(width*2), ticks, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    leg = plt.legend(bars, ('1st Layer', '2nd Layer', '3rd Layer', '4th Layer', '5th Layer'), fontsize=14, framealpha=1.0) 
    leg.get_frame().set_edgecolor('black')
    if task == "class" or task == "cls":
        plt.title(f"Correlation of EEG RSM with Model RSMs: Recognition Task", fontsize=title_font_size)
    else:
        plt.title(f"Correlation of EEG RSM with Model RSMs: Grasp Task", fontsize=title_font_size)
    plt.xlabel("Time Periods (ms from Stimulus Onset)", fontsize=axis_font_size)
    if corr_type=="kendall": plt.ylabel("Kendall's T", fontsize=axis_font_size)
    elif corr_type=="pearson": plt.ylabel("Pearson Correlation (r-value)", fontsize=axis_font_size)
    elif corr_type=="spearman": plt.ylabel("Spearman's Rank \nCorrelation Coefficient", fontsize=axis_font_size)
    ax = plt.gca()
    ax.set_ylim([0, 0.4])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.figtext(0.1, 0.02, "*: p-value < 0.05 \n **: p-value < 0.01 ", ha="center", fontsize=13)
    plt.savefig("vis/rsm_correlation_1/%s_%s_%s" % (task, name_suffix, corr_type))
suffix = sys.argv[1]
corr_type = sys.argv[2]
try: bad_participants = [int(re.sub(r'\D','', num)) for num in suffix.split('-')]
except: bad_participants = []
desire_times = [(0, 50), (50, 100),(100, 150),(150, 200),(200,250), (250, 400)]  
#timepoints are identical across the files so this can stay the same
tp_file = 'data/timepoints_8_cls.csv'
timepoints = np.loadtxt(tp_file, delimiter=',')
times = [(min(timepoints, key=lambda x:abs(x-tp[0])), min(timepoints, key=lambda x:abs(x-tp[1]))) for tp in desire_times]
time_eeg_rsm("grasp")
time_eeg_rsm("class")
exit()
task = sys.argv[3]
try: 
    model_path = sys.argv[4]
except:
    model_path = ""

# for i in range(1,21):
#     if i in [2, 10,13,17]: continue

#change these as necessary
#desire_times = [(0, 50), (50, 75),(75, 125),(125, 166),(166,205), (205, 253), (253, 300)]  
#desire_times = [(0, 100), (100, 200),(200, 300),(300, 400)] 
#desire_times = [(0, 75), (75, 150),(150, 225),(225, 300), (300, 375)] 
desire_times = [(-25,0),(0, 25),(25,50), (50, 75),(75,100),(100, 125),(125,150),(150, 175),(175,200), (200,225),(225,250),(250,275),(275,300), (300,325), (325,350), (350,375),(375,400)]
#desire_times = [(-50,0), (0, 50), (50, 100),(100, 150),(150, 200),(200,250), (250, 300), (300,350), (350, 400) ]  
#timepoints are identical across the files so this can stay the same
tp_file = 'data/timepoints_8_cls.csv'
timepoints = np.loadtxt(tp_file, delimiter=',') 
#convert desired timepoints into actual time points
times = [(min(timepoints, key=lambda x:abs(x-tp[0])), min(timepoints, key=lambda x:abs(x-tp[1]))) for tp in desire_times]
if model_path == "depth":
    model_rsm_folder_path = f'saved_model_rsms/{model_path}'
elif model_path: 
    model_rsm_folder_path = f'saved_model_rsms/{task}/{model_path}'
else:
    model_rsm_folder_path = f'saved_model_rsms/'
    
comparative_analysis(model_rsm_folder_path, timepoints, times, task=task, name_suffix=suffix, corr_type=corr_type, single = -1, bad_participants=bad_participants)
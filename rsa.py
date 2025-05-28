import enum
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
from process_data import get_data_matlab
import sys
import os
import sys
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

def perform_rsm_vis(times, task="cls"):
    for i in range(3):
        data = get_data_matlab(task)
        labels = data.keys()
        activations_flat = []
        timepoints = np.loadtxt('data/timepoints_8_%s.csv' % task, delimiter=',') 
        
        time_period = (np.where(timepoints == times[i][0])[0][0], np.where(timepoints == times[i][1])[0][0])
        points_per_object = {}
        for cat in labels:
            points_per_object[cat] = 0
            for object_data in data[cat]:
                for eeg in object_data:
                    relevant_signal = eeg[time_period[0]:time_period[1], :]
                    activations_flat.append(relevant_signal.flatten())
                    points_per_object[cat] += 1
        act_array = np.asarray(activations_flat)
        result = squareform(pdist(act_array, metric="correlation"))
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
        plt.title("%sms to %sms grasp (correlation)" % (times[i][0], times[i][1]))
        plt.savefig('vis/%s/ts%s_correlation.png' % (task,(i+1)))   
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
        plt.title("%sms to %sms Averaged grasp (correlation)" % (times[i][0], times[i][1]))
        plt.savefig('vis/%s/ts%s_correlation_avr.png' % (task, (i+1)))   
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
      
def comparative_analysis(model_rsm_path, timepoints, times, task="cls", name_suffix="plot"):
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
    data = get_data_matlab(task,avr=True)
    
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
        model_order = ["rgb.npy", "features_0.npy", "features_4.npy", "features_7.npy", "features_10.npy"]
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
            corrs[i].append(pearsonr(rsm1_flat, rsm2_flat))
            
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
    width = 0.15
    plt.figure()
    bars = []
    n = 325
    for i in range(5):   
        vals = corrs[:, i, 0]
        # confidence_intervals = [confidence_interval(r, n) for r in corrs[:, i, 0]]
        # print(confidence_intervals)
        # # Extract lower and upper bounds
        # lower_bounds = [ci[0] for ci in confidence_intervals]
        # upper_bounds = [ci[1] for ci in confidence_intervals]
        bar = plt.bar(ind + width*i, vals,  width, color = (0.2, 0.4, 0.2, 1- 0.1*i)) 
        bars.append(bar)
        for j, rect in enumerate(bar):
            height = rect.get_height()
            if corrs[j, i][1] < 0.05:
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, "*", ha='center', va='bottom')
    ticks = [f'{desire_times[i][0]}-{desire_times[i][1]}' for i in range(len(desire_times))]
    plt.xticks(ind+width,ticks) 
    plt.legend(bars, ('1st Layer', '2nd Layer', '3rd Layer', '4th Layer', '5th Layer') ) 
    if task == "class" or task == "cls":
        plt.title(f"Correlation of EEG RSM with Model RSMs: \n Recognition Task - {name_suffix}")
    else:
        plt.title(f"Correlation of EEG RSM with Model RSMs:\n Grasp Task - {name_suffix}")
    plt.xlabel("Time Periods (ms from beginning of stimulus)")
    plt.ylabel("Correlation")
    ax = plt.gca()
    ax.set_ylim([-0.2, 0.5])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.figtext(0.1, 0.01, "*: p-value < 0.05", ha="center", fontsize=10)
    plt.savefig("vis/rsm_correlation/%s_%s" % (task, name_suffix))
suffix = sys.argv[1]
task = sys.argv[2]
desire_times = [(-50, 0), (0, 75), (75, 125),(125, 175),(175, 225),(225,300), (300, 375)]  
#desire_times = [(0, 50), (50, 100),(100, 150),(150, 200),(200,250), (250, 300)]  
#timepoints are identical across the files so this can stay the same
tp_file = 'data/timepoints_8_cls.csv'
timepoints = np.loadtxt(tp_file, delimiter=',')
times = [(min(timepoints, key=lambda x:abs(x-tp[0])), min(timepoints, key=lambda x:abs(x-tp[1]))) for tp in desire_times]
visualize_eeg_rsm(task)
exit()

model_path = sys.argv[3]

for task in [task]:
    #change these as necessary
    desire_times = [(-50, 0), (0, 75), (75, 125),(125, 175),(175, 225),(225,300), (300, 375)]  
    #desire_times = [(0, 50), (50, 100),(100, 150),(150, 200),(200,250), (250, 300)]  
    #timepoints are identical across the files so this can stay the same
    tp_file = 'data/timepoints_8_cls.csv'
    timepoints = np.loadtxt(tp_file, delimiter=',') 
    #convert desired timepoints into actual time points
    times = [(min(timepoints, key=lambda x:abs(x-tp[0])), min(timepoints, key=lambda x:abs(x-tp[1]))) for tp in desire_times]
    model_rsm_folder_path = f'saved_model_rsms/{model_path}'
    comparative_analysis(model_rsm_folder_path, timepoints, times, task, name_suffix=suffix)
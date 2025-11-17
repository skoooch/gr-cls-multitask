import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.parameters import Params

from sklearn.preprocessing import normalize
# Experiment parameters
TYPES = ['cls', 'grasp']
LAYERS = ['first','features.0','features.4', 'features.7', 'features.10']

R = 100.
DELTA = 0.2
DIR = 'shap'

params = Params()

def get_players(directory):
    ## Load the list of all players (filters) else save
    if 'players.txt' in os.listdir(directory):
        players = open(os.path.join(directory, 'players.txt')).read().split(',')
        players = np.array(players)
    else:
        raise Exception("Players do not exist!")

    return players


def instatiate_chosen_players(directory, players):
    if 'chosen_players.txt' not in os.listdir(directory):
        open(os.path.join(directory, 'chosen_players.txt'), 'w').write(','.join(
            np.arange(len(players)).astype(str)))

def get_results_list(directory):
    results = []
    for file in os.listdir(directory):
        if file.endswith('.h5'):
            results.append(os.path.join(directory, file))

    return results

def get_result(h5file):
    with h5py.File(h5file, 'r') as foo:
        return foo['mem_tmc'][:]

def get_shapley_top_k(model_name, layer, k):
    ## CB directory
    run_name = '%s_%s' % (model_name, layer)
    run_dir = os.path.join(DIR, run_name)

    players = get_players(run_dir)
    instatiate_chosen_players(run_dir, players)    
    results = get_results_list(run_dir)

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]

    for result in results:
        mem_tmc = get_result(result)
        sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
        squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
        counts += np.sum(mem_tmc != -1, 0)

    # No. of iterations for each neuron
    counts = np.clip(counts, 1e-12, None)
    # Expected shapley values of each neuron
    vals = sums / (counts + 1e-12)

    sorted_vals_idx = np.argsort(vals)[-k:]
    return sorted_vals_idx

def get_variance_std(sums, vals, squares, counts):
    variances = R * np.ones_like(vals)
    variances[counts > 1] = squares[counts > 1]
    variances[counts > 1] -= (sums[counts > 1] ** 2) / counts[counts > 1]
    variances[counts > 1] /= (counts[counts > 1] - 1)

    stds = variances ** (1/2)

    return variances, stds

def get_cb_bounds(vals, variances, counts):
    # Empriical berstein conf bounds
    cbs = R * np.ones_like(vals)
    cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / DELTA) / counts[counts > 1]) +\
    7/3 * R * np.log(2 / DELTA) / (counts[counts > 1] - 1)

    return cbs

if __name__ == '__main__':
    if DIR not in os.listdir('vis'):
        os.mkdir(os.path.join('vis', DIR))
    top_k = 128
    values = False
    diff = True
    model_name = params.MODEL_NAME
    final = np.zeros((5, top_k, 2))
    for i, layer in enumerate(LAYERS):
        results_dict = {}
        players = []
        for model_type in TYPES:
            ## CB directory
            run_name = '%s_%s_%s' % (model_name, layer, model_type)
            run_dir = os.path.join(DIR, run_name)

            players = get_players(run_dir)
            instatiate_chosen_players(run_dir, players)    
            results_dict[model_type] = get_results_list(run_dir)
        vals = {}
        for task in results_dict.keys():
            squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
            for result in results_dict[task]:
                mem_tmc = get_result(result)
                sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
                squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
                counts += np.sum(mem_tmc != -1, 0)
            # No. of iterations for each neuron
            counts = np.clip(counts, 1e-12, None)
            # Expected shapley values of each neuron
            vals[task] = sums / (counts + 1e-12)
        # Assuming vals is already calculated for the two tasks:
        task_1 = list(results_dict.keys())[0]  # Access the first task
        task_2 = list(results_dict.keys())[1]  # Access the second task
        # Extract the values for both tasks
        vals_task_1 = vals[task_1]
        vals_task_2 = vals[task_2]
        if values:
            final[i,:len(vals_task_1),0] = vals_task_1
            final[i,:len(vals_task_2),1] = vals_task_2
        else:
            if i == 0: top_k = 128
            elif i == 1: top_k = 32
            if diff:
                vals_task_1_array = normalize(np.array(vals_task_1)[:,np.newaxis], axis=0).ravel()
                vals_task_2_array = normalize(np.array(vals_task_2)[:,np.newaxis], axis=0).ravel()
                vals_task_1 = vals_task_1_array - vals_task_2_array
                vals_task_2 = vals_task_2_array - vals_task_1_array
            ind_task_1 = np.argpartition(vals_task_1, -top_k)[-top_k:]
            ind_task_2 = np.argpartition(vals_task_2, -top_k)[-top_k:]
            ind_task_1 = np.flip(ind_task_1[np.argsort(vals_task_1[ind_task_1])])
            ind_task_2 = np.flip(ind_task_2[np.argsort(vals_task_2[ind_task_2])])
            final[i,:len(ind_task_1),0] = ind_task_1
            final[i,:len(ind_task_2),1] = ind_task_2
            top_k = 64

        
        # ind_task_1 = np.argpartition(vals_task_1, -top_k)[-top_k:]
        # ind_task_2 = np.argpartition(vals_task_2, -top_k)[-top_k:]
        # ind_task_1 = np.flip(ind_task_1[np.argsort(vals_task_1[ind_task_1])])
        # ind_task_2 = np.flip(ind_task_2[np.argsort(vals_task_2[ind_task_2])])
    
    if values:
        np.save(f'shap_arrays/shap_values.npy', final)
    else:
        if not diff: np.save(f'shap_arrays/sort_shap_indices_depth.npy', final)
        else: np.save(f'shap_arrays/sort_shap_indices_diff_depth.npy', final)
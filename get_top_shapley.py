import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.parameters import Params
from scipy import stats
import shutil
from sklearn.preprocessing import normalize
# Experiment parameters
TYPES = ['cls', 'grasp']
LAYERS = ['rgb_features.0','features.0','features.4', 'features.7', 'features.10']

R = 100.
DELTA = 0.2

DIR = 'shap'
from scipy.stats import norm

def fisher_z_transform(r):
    """Perform Fisher's Z-transformation."""
    return np.arctanh(r)

def inverse_fisher_z_transform(z):
    """Convert back from Fisher's Z-transformation to correlation."""
    return np.tanh(z)

def confidence_interval(r, n, alpha=0.05):
    """Compute the confidence interval for a correlation coefficient."""
    # Fisher Z transformation of r
    z = fisher_z_transform(r)
    
    # Standard error of Z
    se = 1 / np.sqrt(n - 3)
    
    # Z critical value for the desired confidence level
    z_critical = norm.ppf(1 - alpha / 2)
    
    # Confidence interval in Z scale
    z_conf_interval = z_critical * se
    z_lower = z - z_conf_interval
    z_upper = z + z_conf_interval
    
    # Convert back to r scale
    r_lower = inverse_fisher_z_transform(z_lower)
    r_upper = inverse_fisher_z_transform(z_upper)
    
    return r_lower, r_upper

params = Params()
def get_r(players, results_dict, layer):
    """
    (Scatter plot)
    (Data point = kernel w/ coordinate being (Shapley-value on cls, shapley-value on grasp))
    Plot shapley value scatter plot and calculate correlation
    """
    vals = {}
    for task in results_dict.keys():
        squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
        for result in results_dict[task]:
            mem_tmc = get_result(result)
            print(mem_tmc)
            sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
            squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
            counts += np.sum(mem_tmc != -1, 0)
        # No. of iterations for each neuron
        counts = np.clip(counts, 1e-12, None)
        print(counts)
        print(sums)
        # Expected shapley values of each neuron
        vals[task] = sums / (counts + 1e-12)
    # Assuming vals is already calculated for the two tasks:
    task_1 = list(results_dict.keys())[0]  # Access the first task
    task_2 = list(results_dict.keys())[1]  # Access the second task


    # Extract the values for both tasks
    vals_task_1 = vals[task_1]
    vals_task_2 = vals[task_2]
    vals_task_1 = np.array(vals_task_1)
    vals_task_2 = np.array(vals_task_2)
    return stats.pearsonr(vals_task_1, vals_task_2)
    
def plot_layer_by_task(players, results_dict, layer):
    """
    (Scatter plot)
    (Data point = kernel w/ coordinate being (Shapley-value on cls, shapley-value on grasp))
    Plot shapley value scatter plot and calculate correlation
    """
    vals = {}
    for task in results_dict.keys():
        squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
        for result in results_dict[task]:
            mem_tmc = get_result(result)
            print(mem_tmc)
            sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
            squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
            counts += np.sum(mem_tmc != -1, 0)
        # No. of iterations for each neuron
        counts = np.clip(counts, 1e-12, None)
        print(counts)
        print(sums)
        # Expected shapley values of each neuron
        vals[task] = sums / (counts + 1e-12)
    # Assuming vals is already calculated for the two tasks:
    task_1 = list(results_dict.keys())[0]  # Access the first task
    task_2 = list(results_dict.keys())[1]  # Access the second task


    # Extract the values for both tasks
    vals_task_1 = vals[task_1]
    vals_task_2 = vals[task_2]
    vals_task_1 = np.array(vals_task_1)
    vals_task_2 = np.array(vals_task_2)
    r,_ = stats.pearsonr(vals_task_1, vals_task_2)
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(vals_task_1, vals_task_2, color='b', alpha=0.7)
    plt.title(f"Shapley Values of {task_1} vs {task_2} on {layer}")
    plt.xlabel(f"Shapley Values for {task_1}")
    plt.ylabel(f"Shapley Values for {task_2}")
    plt.grid(True)
    plt.text(0.05, 0.95, f"r = {r:.4f}", transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.savefig('vis/shap/layer_corr/layer_corr_%s.png' % layer)

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


def plot_shapley_dist(players, results, model_type, layer):
    """
    (Bar chart)
    (Shapley-values against Kernel idx)
    Plot shapley value bar chart with variance and conf bounds
    """
    # Create saving directory
    if 'shapley_dist' not in os.listdir('vis/shap'):
        os.mkdir(os.path.join('vis/shap', 'shapley_dist'))

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
    for result in results:
        mem_tmc = get_result(result)
        print(mem_tmc)
        sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
        squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
        counts += np.sum(mem_tmc != -1, 0)
    # No. of iterations for each neuron
    counts = np.clip(counts, 1e-12, None)
    print(counts)
    print(sums)
    # Expected shapley values of each neuron
    vals = sums / (counts + 1e-12)
    print(vals)
    # Variance of shapley values of each neuron
    variances, stds = get_variance_std(sums, vals, squares, counts)
    print(variances)
    # Empirical berstein confidence bounds for each neuron
    cbs = get_cb_bounds(vals, variances, counts)
    sorted_vals_idx = np.argsort(vals)[-params.TOP_K:]
    top_k_vals = np.zeros((len(vals)))
    top_k_vals[sorted_vals_idx] = 1

    colors = np.where(top_k_vals == 1, 'coral', 'turquoise')
    for i in range(len(players)):
        if top_k_vals[i] == 1:
            shutil.copy('features/%s/%s.png' % (layer, i), 'features/%s_%s_%s.png' % (layer, model_type, i))
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(vals)), vals, yerr=cbs, align='center', ecolor='lightgrey', color=colors)
    ax.set_xlabel('Kernel Index')
    ax.set_ylabel('Shapley Scores')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    fig.suptitle("Shapley Values for %s on %s" % (layer, model_type))

    plt.savefig('vis/shap/shapley_dist/shapley_dist_%s_%s.png' % (model_type, '_'.join(layer.split('.'))))


def plot_shapley_conf_trend(players, results, model_type, layer):
    """
    (Line graph)
    (Variance/confBound against iteration)
    Plot variance / confidence bounds line graph against No. of iters
    """
    # Create saving directory
    if 'shapley_confidence_bounds' not in os.listdir('vis/shap'):
        os.mkdir(os.path.join('vis/shap', 'shapley_confidence_bounds'))

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]

    iter = 0
    cbs_history = np.zeros((1, len(players)))
    for result in results:
        mem_tmc = get_result(result)
        for mem_tmc_instance in mem_tmc:
            sums += (mem_tmc_instance != -1) * mem_tmc_instance
            squares += (mem_tmc_instance != -1) * (mem_tmc_instance ** 2)
            counts += mem_tmc_instance != -1
            iter += 1
    
            # No. of iterations for each neuron
            counts = np.clip(counts, 1e-12, None)
            # Expected shapley values of each neuron
            vals = sums / (counts + 1e-12)
            # Variance of shapley values of each neuron
            variances, stds = get_variance_std(sums, vals, squares, counts)
            # Empirical berstein confidence bounds for each neuron
            cbs = get_cb_bounds(vals, variances, counts)
            cbs_history = np.concatenate((cbs_history, np.expand_dims(cbs, 0)), axis=0)

    # Remove initial cbs_history array (zero array)
    cbs_history = cbs_history[1:, :]

    sorted_vals_idx = np.argsort(vals)[-params.TOP_K:]
    top_k_vals = np.zeros((len(vals)))
    top_k_vals[sorted_vals_idx] = 1

    fix, ax = plt.subplots()
    iter_axis = np.arange((cbs_history.shape[0]))
    for kernel_idx in sorted_vals_idx:
        ax.plot(iter_axis, cbs_history[:, kernel_idx], label='Kernel %s' % kernel_idx)

    ax.plot(iter_axis, np.repeat(np.min(cbs_history), len(iter_axis)), '-.', linewidth=.5, color='red')
    
    ax.legend()
    plt.ylim([0, 25])
    plt.yticks(list(plt.yticks()[0]) + [np.min(cbs_history)])
    
    plt.savefig('vis/shap/shapley_confidence_bounds/shapley_confidence_bounds_%s_%s.png' % (model_type, '_'.join(layer.split('.'))))


if __name__ == '__main__':
    if DIR not in os.listdir('vis'):
        os.mkdir(os.path.join('vis', DIR))
    top_k = 64
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
        # vals_task_1_array = normalize(np.array(vals_task_1)[:,np.newaxis], axis=0).ravel()
        # vals_task_2_array = normalize(np.array(vals_task_2)[:,np.newaxis], axis=0).ravel()
        # vals_task_1 = vals_task_1_array - vals_task_2_array
        # vals_task_2 = vals_task_2_array - vals_task_1_array
        if i == 1: top_k=32
        # ind_task_1 = np.argpartition(vals_task_1, -top_k)[-top_k:]
        # ind_task_2 = np.argpartition(vals_task_2, -top_k)[-top_k:]
        # ind_task_1 = np.flip(ind_task_1[np.argsort(vals_task_1[ind_task_1])])
        # ind_task_2 = np.flip(ind_task_2[np.argsort(vals_task_2[ind_task_2])])
        final[i,:len(vals_task_1),0] = vals_task_1
        final[i,:len(vals_task_2),1] = vals_task_2
        top_k = 64
    np.save(f'shap_values.npy', final)
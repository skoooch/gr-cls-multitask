import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.parameters import Params
from scipy import stats
import shutil
# Experiment parameters
TYPES = ['cls', 'grasp']
LAYERS = ['first','features.0','features.4', 'features.7', 'features.10']


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
    vals_task_1 = np.array(vals_task_1)
    vals_task_2 = np.array(vals_task_2)
    # Bootstrapped Pearson correlation
    n_bootstrap = 1000
    rng = np.random.default_rng(seed=42)
    r_bootstrap = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(vals_task_1), len(vals_task_1))
        r, _ = stats.pearsonr(vals_task_1[idx], vals_task_2[idx])
        r_bootstrap.append(r)
    r_bootstrap = np.array(r_bootstrap)
    r_mean = np.mean(r_bootstrap)
    r_ci_lower = np.percentile(r_bootstrap, 2.5)
    r_ci_upper = np.percentile(r_bootstrap, 97.5)
    return r_mean, (r_ci_lower, r_ci_upper)
    
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
    vals_task_1 = np.array(vals_task_1)
    # Remove value at index 36 from both arrays
    vals_task_2 = np.array(vals_task_2)
    
    vals_task_1 = np.delete(vals_task_1, 37)
    vals_task_2 = np.delete(vals_task_2, 37)
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
    exit()

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
        
        sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
        squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
        counts += np.sum(mem_tmc != -1, 0)
    # No. of iterations for each neuron
    counts = np.clip(counts, 1e-12, None)
   

    # Expected shapley values of each neuron
    vals = sums / (counts + 1e-12)
    

    # Variance of shapley values of each neuron
    variances, stds = get_variance_std(sums, vals, squares, counts)
   

    # Empirical berstein confidence bounds for each neuron
    cbs = get_cb_bounds(vals, variances, counts)
    sorted_vals_idx = np.argsort(vals)[-params.TOP_K:]
    top_k_vals = np.zeros((len(vals)))
    top_k_vals[sorted_vals_idx] = 1

    # Set bar colors
    colors = np.where(top_k_vals == 1, 'coral', 'turquoise')
    if model_type == 'cls':
        colors = np.where(top_k_vals == 1, 'red', 'lightcoral')
    elif model_type == 'grasp':
        colors = np.where(top_k_vals == 1, 'blue', 'lightblue')
    else: # fallback color
        colors = np.where(top_k_vals == 1, 'grey', 'lightgrey')

    # for i in range(len(players)):
    #     if top_k_vals[i] == 1:
    #         shutil.copy('features/%s/%s.png' % (layer, i), 'features/%s_%s_%s.png' % (layer, model_type, i))
            
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

def plot_all_layer_scatter(players_dict, results_dict_by_layer, layers):
    '''
    plot correlation in neuron shapley values between tasks across layers
    plot shapley graphs for each layer under the correlation graph
    '''

    num_layers = len(layers)
    r_values = []
    scatter_data = []
    confidence_intervals = []
    # data for each scatter plot
    for layer in layers:
        players = players_dict[layer]
        results_dict = results_dict_by_layer[layer]

        vals = {}
        for task in results_dict:
            squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
            for result in results_dict[task]:
                mem_tmc = get_result(result)
                sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
                squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
                counts += np.sum(mem_tmc != -1, 0)
            counts = np.clip(counts, 1e-12, None)
            vals[task] = sums / (counts + 1e-12)
        task_1, task_2 = list(vals.keys())
        x = vals[task_1]
        y = vals[task_2]
        print(len(x))
        r, p = get_r(players, results_dict, layer)
        r_values.append(r)
        confidence_intervals.append(p)
        scatter_data.append((x, y, r, layer, task_1, task_2))

    # plot main correlation line
    fig, ax = plt.subplots(figsize=(10, 8))
    x_values = np.arange(1, len(r_values) + 1)
    n = 64 # can change number neurons in each layer

    # confidence_intervals = [confidence_interval(r, n) for r in r_values]
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]

    ax.errorbar(x_values, r_values, 
                yerr=[np.abs(np.array(r_values) - np.array(lower_bounds)), 
                      np.abs(np.array(upper_bounds) - np.array(r_values))],
                fmt='o', capsize=5, linestyle='--', color="black")

    ax.set_xticks(x_values)
    ax.set_xticklabels([str(i) for i in x_values])
    ax.set_ylabel('Correlation (r-value)', labelpad=2)
    ax.set_xlabel('Convolutional layer', labelpad=2)
    ax.set_title('Correlation in Neuron Shapley Values Between Tasks Across Layers')

    # add inset axes below x-axis
    fig.subplots_adjust(bottom=0.3)
    for i, (x, y, r, layer, task_1, task_2) in enumerate(scatter_data):
        inset_width = 0.12
        inset_height = 0.18
        left = 0.13 + i * (0.83 / len(scatter_data))  # even horizontal spacing
        bottom = 0.05

        inset_ax = fig.add_axes([left, bottom, inset_width, inset_height])
        inset_ax.scatter(x, y, alpha=0.7, color="black")
        inset_ax.set_title(f'r = {r:.2f}', fontsize=8)
        inset_ax.set_xlabel(f'{task_1}', fontsize=6, labelpad=1)
        inset_ax.set_ylabel(f'{task_2}', fontsize=6, labelpad=1)
        inset_ax.tick_params(axis='both', which='major', labelsize=6)

    plt.savefig('vis/shap/layer_corr/combined_below_graph_black.png', dpi=300)

if __name__ == '__main__':
    if DIR not in os.listdir('vis'):
        os.mkdir(os.path.join('vis', DIR))
    model_name = params.MODEL_NAME

    players_dict = {}
    results_dict_by_layer = {}
    
    for layer in LAYERS:
        results = {}
        players = []
        for model_type in TYPES:
            ## CB directory
            run_name = '%s_%s_%s' % (model_name, layer, model_type)
            run_dir = os.path.join(DIR, run_name)

            players = get_players(run_dir)
            instatiate_chosen_players(run_dir, players)    
            results[model_type] = get_results_list(run_dir)
            players_dict[layer] = players
            results_dict_by_layer[layer] = results

        #plot_layer_by_task(players, results, layer)

    for model_type in TYPES:
        for layer in LAYERS:
            ## CB directory
            run_name = '%s_%s_%s' % (model_name, layer, model_type)
            run_dir = os.path.join(DIR, run_name)

            players = get_players(run_dir)
            instatiate_chosen_players(run_dir, players)    
            results = get_results_list(run_dir)
           
            #plot_shapley_dist(players, results, model_type, layer)
    
    #         plot_shapley_conf_trend(players, results, model_type, layer)
    r_value = []
    confidence_intervals = []
    for layer in LAYERS:
        results = {}
        players = []
        for model_type in TYPES:
            ## CB directory
            run_name = '%s_%s_%s' % (model_name, layer, model_type)
            run_dir = os.path.join(DIR, run_name)

            players = get_players(run_dir)
            instatiate_chosen_players(run_dir, players)    
            results[model_type] = get_results_list(run_dir)
        #plot_layer_by_task(players, results, layer)
        r,p = get_r(players, results, layer)
        confidence_intervals.append(p)
        r_value.append(r)
    # n = 64  # Assuming sample size of 30

    # # Calculate confidence intervals for each r-value
    # #confidence_intervals = [confidence_interval(r, n) for r in r_value]

    # # Extract lower and upper bounds
    # lower_bounds = [ci[0] for ci in confidence_intervals]
    # upper_bounds = [ci[1] for ci in confidence_intervals]

    # # Plotting
    # fig, ax = plt.subplots()
    # x_values = np.arange(1, len(r_value) + 1)

    # # Plot the r values with confidence intervals
    # ax.errorbar(x_values, r_value, 
    #             yerr=[np.abs(np.array(r_value) - np.array(lower_bounds)), 
    #                 np.abs(np.array(upper_bounds) - np.array(r_value))],
    #             fmt='o', capsize=5, linestyle='--')

    # # Customize the plot
    # ax.set_xticks(x_values)
    # ax.set_xticklabels([f'{i}' for i in range(1, len(r_value) + 1)])
    # ax.set_ylabel('Correlation (r-value)')
    # ax.set_xlabel('Convolutional layer')
    # ax.set_title('Correlation in Neuron Shapley Values Between Tasks Across Layers')
    # ax.legend()
    # plt.savefig('vis/shap/layer_corr/all_layers_boot.png')

    plot_all_layer_scatter(players_dict, results_dict_by_layer, LAYERS)

    

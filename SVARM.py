
import numpy as np

from utils.experiment_util import normalize_shapley_value
import sys
import torch
import torch.nn as nn
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v2 import DataLoader
import math
from utils.parameters import Params
import os
from shapley_debug_connections import get_model
from utils.utils import get_correct_preds, get_acc, get_correct_cls_preds_from_map
from utils.grasp_utils import get_correct_grasp_preds
import time
from training.single_task.evaluation import denormalize_grasp, map2singlegrasp  
params = Params()
LAYERS = ['first','features.0','features.4', 'features.7', 'features.10']
SIZES = [128,32,64,64,64]
TASK = 'grasp'
layer_i = 1
LAYER = LAYERS[layer_i]
METRIC = 'accuracy'
TRUNCATION_ACC = 50.
DEVICE = sys.argv[1]
PARALLEL_ID = sys.argv[2]
DIR = 'shap/connections/'
MODEL_NAME = params.MODEL_NAME
MODEL_PATH = params.MODEL_WEIGHT_PATH

def get_accuracy(model, shap_mask=[], activations=None, task='cls', device='cuda:0', subset=None):
    if task == 'cls':
        return get_cls_acc(model, subset)[0]
    elif task == 'grasp':
        return get_grasp_acc(model, subset)[0]
    else:
        raise ValueError('Invalid task!')
def get_cls_acc(model, subset, batch_size=5):
    """Returns the test accuracy and loss of a CLS model."""
    loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
      for (img, cls_map, label) in subset:
        output = model(img, is_grasp=False)
        batch_correct, batch_total = get_correct_cls_preds_from_map(output, label)
        correct += batch_correct
        total += batch_total
    accuracy = get_acc(correct, total)
    return accuracy, round(loss / total, 3)
def get_grasp_acc(model, subset, batch_size=5):
    """Returns the test accuracy and loss of a Grasp model."""
    loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
      for (img, map, candidates) in subset:
        output = model(img, is_grasp=True)
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
    accuracy = get_acc(correct, total)
    return accuracy, round(loss / total, 3)
  
def restore_connections(layer_module, orig_weights):

    layer_module.weight.data.copy_(orig_weights)
    
def remove_connections(model: nn.Module, layer: str, removed_connections: list) -> nn.Module:
    """
    Efficiently silence the impact of specific connections (src_kernel, tgt_kernel) in <layer> of <model>.
    """
    with torch.no_grad():
        # Directly access the layer's weight
        layer_module = dict(model.named_modules())[layer]
        W = layer_module.weight

        # Store original weights
        orig_weights = W.data.clone()
        if removed_connections:
            src_idx = torch.tensor([src for src, tgt in removed_connections], dtype=torch.long)
            tgt_idx = torch.tensor([tgt for src, tgt in removed_connections], dtype=torch.long)
            W.data[tgt_idx, src_idx, :, :] = 0
        W = orig_weights
    return model, orig_weights

def get_players(src_kernels, tgt_kernels):
    ## Load the list of all players (filters) else save
    players = []
    for j in src_kernels:
        for i in tgt_kernels:
            players.append((i,j))
    return players

def convert_index_to_value(idx):
    src = idx // SIZES[layer_i+1]
    tgt = idx % SIZES[layer_i+1]
    return (src, tgt)

# SVARM
class SVARM():
  def __init__(self, normalize=False, warm_up=False, layer_i = 1, task = 'grasp', budget = 100, alpha=0.05,
               stop_eps=None, progress_every=50, skip=10):
    self.normalize = normalize
    self.warm_up = warm_up
    self.layer_i = layer_i
    self.initial_budget = budget
    self.budget = budget
    self.alpha = alpha                # For CI (two‑sided)
    self.stop_eps = stop_eps          # Desired max half‑width (stopping tolerance); None => no early stop
    self.progress_every = progress_every
    self.skip=skip
    self.grand_co_value = [85, 81.5][task=='grasp']
    shap_values = np.load("shap_arrays/shap_values.npy")
    shap_values[:,:,0] /= 85
    shap_values[:,:,1] /= 81.5
    # Get indices of top 5 values along the last axis
    shap_values = np.mean(shap_values, axis=-1)

    top5_indices = np.argsort(shap_values, axis=-1)[..., -5:]
    top_5_keys = top5_indices[layer_i,:]
    players_tuple = get_players(range(SIZES[layer_i]), range(SIZES[layer_i+1]))
    self.connections = players_tuple
    self.players = np.array([i for i in range(len(self.connections))])
    self.n = len(self.players)
    self.player_steps = [0] * self.n
    
    self.shapley_values = np.zeros(self.n)
    self.shapley_mean_values = np.zeros(self.n)
    self.shapley_mean_square_values = np.zeros(self.n)
    self.shapley_var_values = np.zeros(self.n)
    self.idx_to_val = {}
    for p in self.players:
        self.idx_to_val[p] = convert_index_to_value(p)
    self.task = task
    self.model = get_model(MODEL_PATH, DEVICE)
    self.subset = []
    #running stats (Welford) for plus / minus samples per player
    self.plus_mean  = np.zeros(self.n)
    self.plus_M2    = np.zeros(self.n)
    self.plus_n     = np.zeros(self.n, dtype=int)
    self.minus_mean = np.zeros(self.n)
    self.minus_M2   = np.zeros(self.n)
    self.minus_n    = np.zeros(self.n, dtype=int)

    # Cached CI results
    self.ci_low  = np.full(self.n, -np.inf)
    self.ci_high = np.full(self.n,  np.inf)
    self.ci_half = np.full(self.n,  np.inf)
    
  def approximate_shapley_values(self, mse_target=1e-4, log_dir=None) -> dict:
    self.phi_i_plus = np.zeros(self.n)
    self.phi_i_minus = np.zeros(self.n)
    self.c_i_plus = np.zeros(self.n)
    self.c_i_minus = np.zeros(self.n)
    self.H_n = sum([1/s for s in range(1, self.n+1)])
    data_loader = DataLoader(params.TEST_PATH, 25, params.TRAIN_VAL_SPLIT)
    if self.task == 'grasp':
      for i, (img, map, candidates) in enumerate(data_loader.load_grasp()):
        
        if i >=5: break
        self.subset.append((img, map, candidates))
    else:
      for i, (img, map, candidates) in enumerate(data_loader.load_cls()):
        if i >=5: break
        self.subset.append((img, map, candidates))
    if self.warm_up:
      self.__conduct_warmup()

    more_budget = True
    iter_counter = 0
    while more_budget:
      A_plus = self.__sample_A_plus()
      more_budget = self.__positive_update(A_plus)
      if not more_budget:
        break

      A_minus = self.__sample_A_minus()
      more_budget = self.__negative_update(A_minus)
      iter_counter += 1
      if (self.progress_every and iter_counter % (self.progress_every * 1000) == 0):
        self._update_all_cis()
        # NEW integrated logging
        self._log_progress(iter_counter, mse_target=mse_target, log_dir=log_dir)
    self._update_all_cis()
    return {
      'shapley': self.get_estimates(),
      'ci_low': self.ci_low,
      'ci_high': self.ci_high,
      'ci_halfwidth': self.ci_half,
      'plus_counts': self.plus_n,
      'minus_counts': self.minus_n
    }
  def _welford_update_plus(self, idx, value):
    n = self.plus_n[idx] + 1
    delta = value - self.plus_mean[idx]
    self.plus_mean[idx] += delta / n
    delta2 = value - self.plus_mean[idx]
    self.plus_M2[idx] += delta * delta2
    self.plus_n[idx] = n

  def _welford_update_minus(self, idx, value):
    n = self.minus_n[idx] + 1
    delta = value - self.minus_mean[idx]
    self.minus_mean[idx] += delta / n
    delta2 = value - self.minus_mean[idx]
    self.minus_M2[idx] += delta * delta2
    self.minus_n[idx] = n

  def _var_component(self, mean_arr, M2_arr, n_arr):
    var = np.zeros_like(mean_arr)
    mask = n_arr > 1
    var[mask] = M2_arr[mask] / (n_arr[mask] - 1)
    var[~mask] = np.inf
    return var

  def _update_all_cis(self):
    var_plus  = self._var_component(self.plus_mean,  self.plus_M2,  self.plus_n)
    var_minus = self._var_component(self.minus_mean, self.minus_M2, self.minus_n)
    # Shapley estimate (using running coalition means) matches phi_i_plus - phi_i_minus idea
    self.shapley_values = self.plus_mean - self.minus_mean

    # Approx variance of difference (assume independence for approximation)
    with np.errstate(divide='ignore', invalid='ignore'):
      term_plus  = np.where(self.plus_n  > 0, var_plus  / self.plus_n,  np.inf)
      term_minus = np.where(self.minus_n > 0, var_minus / self.minus_n, np.inf)
      var_phi = term_plus + term_minus

    z = self._z_value()
    halfwidth = z * np.sqrt(var_phi)
    self.ci_half = halfwidth
    self.ci_low  = self.shapley_values - halfwidth
    self.ci_high = self.shapley_values + halfwidth

  def _z_value(self):
    # Two-sided z for large-sample normal approx
    # For small samples you could use t; here keep it simple
    return 1.96 if self.alpha == 0.05 else \
           float(np.abs(np.round(np.sqrt(2)*math.erfcinv(self.alpha))))  # fallback

  def _all_converged(self):
    if self.stop_eps is None: return False
    finite_hw = self.ci_half[np.isfinite(self.ci_half)]
    if finite_hw.size == 0: return False
    return np.all(finite_hw <= self.stop_eps)
  def get_all_players(self):
      return self.players

  def __sample_A_plus(self):
    s_plus = np.random.choice(range(1, self.n+1), 1, p=[1/(s*self.H_n) for s in range(1, self.n+1)])
    return np.random.choice(self.get_all_players(), s_plus, replace=False)


  def __sample_A_minus(self):
    s_minus = np.random.choice(range(0, self.n), 1, p=[1/((self.n-s)*self.H_n) for s in range(0, self.n)])
    return np.random.choice(self.get_all_players(), s_minus, replace=False)


  def __positive_update(self, A):
    more_budget, value = self.get_game_value(A)
    for i in A:
      self.phi_i_plus[i] = (self.phi_i_plus[i]*self.c_i_plus[i] + value) / (self.c_i_plus[i] + 1)
      self.c_i_plus[i] += 1
      self._welford_update_plus(i, value)
    return more_budget


  def __negative_update(self, A):
    more_budget, value = self.get_game_value(A)
    players = [i for i in self.get_all_players() if i not in A]
    for i in players:
      self.phi_i_minus[i] = (self.phi_i_minus[i]*self.c_i_minus[i] + value) /(self.c_i_minus[i] + 1)
      self.c_i_minus[i] += 1
      self._welford_update_minus(i, value)
    return more_budget


  def __conduct_warmup(self):
    for i in self.get_all_players():
      players_without_i = [j for j in self.get_all_players() if j != i]

      # sample A_plus
      size_of_A_plus = np.random.choice(self.n, 1)
      A_plus = np.random.choice(players_without_i, size_of_A_plus, replace=False)

      # sample A_minus
      size_of_A_minus = np.random.choice(self.n, 1)
      A_minus = np.random.choice(players_without_i, size_of_A_minus, replace=False)

      # set values
      _, value = self.get_game_value(np.append(A_plus, i))
      self.phi_i_plus[i] = value
      self.c_i_plus[i] = 1

      _, value = self.get_game_value(A_minus)
      self.phi_i_minus[i] = value
      self.c_i_minus[i] = 1

  def get_game_value(self, players_keep):
    if len(players_keep) == 0:
        return True, 0
    else:
        self.budget -= 1
        # Find connections to remove: those in self.players but not in players_keep
        removed_connections = [self.idx_to_val[conn] for conn in self.players if conn not in players_keep]
        # Remove those connections from the model
        self.model, orig_weights = remove_connections(self.model, LAYERS[self.layer_i + 1], removed_connections)
        # Compute the game value (e.g., accuracy or other metric)
        value = get_accuracy(self.model, task=self.task, device = DEVICE, subset = self.subset)
        # Restore original weights
        restore_connections(dict(self.model.named_modules())[LAYERS[self.layer_i+1]], orig_weights)
        return self.budget > 0, value

  def get_estimates(self):
    if not np.any(np.isfinite(self.ci_half)):  # ensure CI computed at least once
      self._update_all_cis()
    est = self.shapley_values  # already maintained in _update_all_cis
    if self.normalize:
      return normalize_shapley_value(est, self.grand_co_value)
    return est
  
  def update_shapley_value(self, player, estimate):
    step = self.player_steps[player]
    self.shapley_values[player] = (self.shapley_values[player] * step + estimate) / (step+1)
    self.player_steps[player] += 1

  def get_name(self) -> str:
    if self.normalize:
      if self.warm_up:
        return 'SVARM_warmup_nor'
      return 'SVARM_nor'
    if self.warm_up:
      return 'SVARM_warmup'
    return 'SVARM'
  BOUND_CONST = 2 * 35.56  # K in your bound

  def get_component_variances(self):
      var_plus  = np.full(self.n, np.inf)
      okp = self.plus_n > 1
      var_plus[okp] = self.plus_M2[okp] / (self.plus_n[okp] - 1)

      var_minus = np.full(self.n, np.inf)
      okm = self.minus_n > 1
      var_minus[okm] = self.minus_M2[okm] / (self.minus_n[okm] - 1)
      return var_plus, var_minus

  def required_T(self, mse_target):
      if np.isscalar(mse_target):
          mse_target_arr = np.full(self.n, mse_target, float)
      else:
          mse_target_arr = np.asarray(mse_target, float)
      var_plus, var_minus = self.get_component_variances()
      num = self.BOUND_CONST * (var_plus + var_minus)
      T_needed = np.full(self.n, np.inf)
      ok = np.isfinite(num) & (mse_target_arr > 0)
      T_needed[ok] = num[ok] / mse_target_arr[ok]
      T_global = np.nanmax(T_needed)
      return T_needed, T_global

  def additional_budget_needed(self, mse_target):
      T_needed, T_global = self.required_T(mse_target)
      T_current = self.initial_budget - self.budget
      extra_pp = np.full(self.n, np.inf)
      finite = np.isfinite(T_needed)
      extra_pp[finite] = np.clip(np.ceil(T_needed[finite] - T_current), 0, None)
      extra_global = np.inf
      if np.isfinite(T_global):
          extra_global = max(0, math.ceil(T_global - T_current))
      return extra_pp, extra_global, T_needed, T_global

  def _log_progress(self, iter_counter, mse_target=1e-4, log_dir=None):
      var_plus, var_minus = self.get_component_variances()
      vp_mean = np.nanmean(var_plus[np.isfinite(var_plus)])
      vm_mean = np.nanmean(var_minus[np.isfinite(var_minus)])
      extra_pp, extra_global, T_needed, T_global = self.additional_budget_needed(mse_target)
      max_ci = np.max(self.ci_half[np.isfinite(self.ci_half)]) if np.any(np.isfinite(self.ci_half)) else np.inf
      msg = (f"[SVARM] it={iter_counter} used={self.initial_budget - self.budget} "
              f"max_CI={max_ci:.4g} vp_mean={vp_mean:.4g} vm_mean={vm_mean:.4g} "
              f"T_global_needed={T_global:.3g} extra_global={extra_global}")
      print(msg)
      # Optional: save snapshot arrays occasionally
      if log_dir:
          hdr = ("time,iter,used,max_ci,vp_mean,vm_mean,T_global,extra_global\n")
          fpath = os.path.join(log_dir, f"svarm_progress{PARALLEL_ID}.csv")
          line = f"{time.time():.3f},{iter_counter},{self.initial_budget - self.budget}," \
                  f"{max_ci},{vp_mean},{vm_mean},{T_global},{extra_global}\n"
          if not os.path.exists(fpath):
              with open(fpath, "w") as f:
                  f.write(hdr)
          with open(fpath, "a") as f:
              f.write(line)
          np.save(os.path.join(log_dir, f'plus_mean_worker{PARALLEL_ID}.npy'),self.plus_mean)
          np.save(os.path.join(log_dir, f'minus_mean_worker{PARALLEL_ID}.npy'),self.minus_mean)
          np.save(os.path.join(log_dir, f'plus_M2_worker{PARALLEL_ID}.npy'),self.plus_M2)
          np.save(os.path.join(log_dir, f'minus_M2_worker{PARALLEL_ID}.npy'),self.minus_M2)
          np.save(os.path.join(log_dir, f'plus_n_worker{PARALLEL_ID}.npy'),self.plus_n)
          np.save(os.path.join(log_dir, f'minus_n_worker{PARALLEL_ID}.npy'),self.minus_n)



if __name__ == "__main__":
    layer_i = int(sys.argv[3])
    task_in = sys.argv[4]
    PARALLEL_ID += f'_{task_in}'
    PARALLEL_ID += f'_{layer_i}'
    estimator = SVARM(normalize=True, warm_up=True, layer_i=layer_i, task=task_in, budget=50000000, alpha=0.05,
            stop_eps=0.01, progress_every=50)
    result = estimator.approximate_shapley_values(mse_target=0.002, log_dir="logs/svarm_run1", )
    print("Shapley:", result['shapley'])
    savedir = 'shap/connections'
    np.save(f'{savedir}/{layer_i}/shapley{PARALLEL_ID}.npy', result['shapley'])
    np.save(f'{savedir}/{layer_i}/ci_low{PARALLEL_ID}.npy', result['ci_low'])
    np.save(f'{savedir}/{layer_i}/ci_high{PARALLEL_ID}.npy', result['ci_high'])
    np.save(f'{savedir}/{layer_i}/ci_halfwidth{PARALLEL_ID}.npy', result['ci_halfwidth'])
    np.save(f'{savedir}/{layer_i}/plus_counts{PARALLEL_ID}.npy', result['plus_counts'])
    np.save(f'{savedir}/{layer_i}/minus_counts{PARALLEL_ID}.npy', result['minus_counts'])
    # ...existing code...
    np.save(f'{savedir}/{layer_i}/plus_mean_worker{PARALLEL_ID}.npy', estimator.plus_mean)
    np.save(f'{savedir}/{layer_i}/minus_mean_worker{PARALLEL_ID}.npy', estimator.minus_mean)
    np.save(f'{savedir}/{layer_i}/plus_M2_worker{PARALLEL_ID}.npy', estimator.plus_M2)
    np.save(f'{savedir}/{layer_i}/minus_M2_worker{PARALLEL_ID}.npy', estimator.minus_M2)
    np.save(f'{savedir}/{layer_i}/plus_n_worker{PARALLEL_ID}.npy', estimator.plus_n)
    np.save(f'{savedir}/{layer_i}/minus_n_worker{PARALLEL_ID}.npy', estimator.minus_n)
    print("CI half-width (max):", np.max(result['ci_halfwidth']))
    
import numpy as np
import sys
layer = sys.argv[1]
task = sys.argv[2]
p = np.load(f'plus_mean_worker1_{task}_{layer}.npy')
m = np.load(f'minus_mean_worker1_{task}_{layer}.npy')
p_M = np.load(f'plus_M2_worker1_{task}_{layer}.npy')
m_M = np.load(f'minus_M2_worker1_{task}_{layer}.npy')
p_n = np.load(f'plus_n_worker1_{task}_{layer}.npy')
m_n = np.load(f'minus_n_worker1_{task}_{layer}.npy')
shap = p-m
var_shap = shap.var()
var_p = p_M / p_n
var_m = m_M / m_n
est_var = var_p /p_n + var_m /m_n
M_emp_t = np.mean(est_var) 
nr = M_emp_t/var_shap
print(f"N = {np.mean(p_n)}")

print(f"Noise ratio = {nr}")
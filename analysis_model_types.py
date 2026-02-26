import numpy as np
import pandas as pd
from scipy import stats

# Read data from CSV
df = pd.read_csv('test_accuracies.csv', header=None, names=['seed', 'cls_acc', 'grasp_acc'])

# Separate depth and RGB models
depth_models = df[df['seed'] < 300]
rgb_models = df[df['seed'] >= 300]

depth_cls = depth_models['cls_acc'].values
depth_grasp = depth_models['grasp_acc'].values
rgb_cls = rgb_models['cls_acc'].values
rgb_grasp = rgb_models['grasp_acc'].values

print(f"Depth models: {len(depth_models)}, RGB models: {len(rgb_models)}\n")

# Test 1: RGB better at classification than depth
t_stat_cls, p_val_cls = stats.ttest_ind(rgb_cls, depth_cls, alternative='greater')
print(f"Classification: RGB > Depth")
print(f"  RGB mean: {np.mean(rgb_cls):.2f}, Depth mean: {np.mean(depth_cls):.2f}")
print(f"  t-statistic: {t_stat_cls:.3f}, p-value: {p_val_cls:.4f}")

# Test 2: RGB worse at grasping than depth
t_stat_grasp, p_val_grasp = stats.ttest_ind(depth_grasp, rgb_grasp,alternative='greater')
print(f"\nGrasping: RGB < Depth")
print(f"  RGB mean: {np.mean(rgb_grasp):.2f}, Depth mean: {np.mean(depth_grasp):.2f}")
print(f"  t-statistic: {t_stat_grasp:.3f}, p-value: {p_val_grasp:.4f}")

# Test 3: Mann-Whitney U (non-parametric alternative)
u_stat_cls, p_val_cls_mw = stats.mannwhitneyu(rgb_cls, depth_cls, alternative='greater')
u_stat_grasp, p_val_grasp_mw = stats.mannwhitneyu(rgb_grasp, depth_grasp, alternative='less')
print(f"\nMann-Whitney U Test:")
print(f"  Classification p-value: {p_val_cls_mw:.4f}")
print(f"  Grasping p-value: {p_val_grasp_mw:.4f}")

# Test 4: Within RGB - classification vs grasping (paired t-test)
t_stat_rgb_paired, p_val_rgb_cls_better = stats.ttest_rel(rgb_cls, rgb_grasp, alternative='greater')
t_stat_rgb_paired2, p_val_rgb_grasp_better = stats.ttest_rel(rgb_grasp, rgb_cls, alternative='greater')
print(f"\n--- Within RGB Models (Paired t-test) ---")
print(f"Classification vs Grasping:")
print(f"  Cls mean: {np.mean(rgb_cls):.2f}, Grasp mean: {np.mean(rgb_grasp):.2f}")
print(f"  Cls > Grasp: t={t_stat_rgb_paired:.3f}, p={p_val_rgb_cls_better:.4f}")
print(f"  Grasp > Cls: t={t_stat_rgb_paired2:.3f}, p={p_val_rgb_grasp_better:.4f}")

# Test 5: Within Depth - classification vs grasping (paired t-test)
t_stat_depth_paired, p_val_depth_cls_better = stats.ttest_rel(depth_cls, depth_grasp, alternative='greater')
t_stat_depth_paired2, p_val_depth_grasp_better = stats.ttest_rel(depth_grasp, depth_cls, alternative='greater')
print(f"\n--- Within Depth Models (Paired t-test) ---")
print(f"Classification vs Grasping:")
print(f"  Cls mean: {np.mean(depth_cls):.2f}, Grasp mean: {np.mean(depth_grasp):.2f}")
print(f"  Cls > Grasp: t={t_stat_depth_paired:.3f}, p={p_val_depth_cls_better:.4f}")
print(f"  Grasp > Cls: t={t_stat_depth_paired2:.3f}, p={p_val_depth_grasp_better:.4f}")

# Significance interpretation
print(f"\n=== Results Summary (α=0.05) ===")
print(f"Between modalities:")
print(f"  RGB better at classification: {'YES' if p_val_cls < 0.05 else 'NO'}")
print(f"  RGB worse at grasping: {'YES' if p_val_grasp < 0.05 else 'NO'}")
print(f"\nWithin RGB:")
print(f"  Classification better than grasping: {'YES' if p_val_rgb_cls_better < 0.05 else 'NO'}")
print(f"  Grasping better than classification: {'YES' if p_val_rgb_grasp_better < 0.05 else 'NO'}")
print(f"\nWithin Depth:")
print(f"  Classification better than grasping: {'YES' if p_val_depth_cls_better < 0.05 else 'NO'}")
print(f"  Grasping better than classification: {'YES' if p_val_depth_grasp_better < 0.05 else 'NO'}")
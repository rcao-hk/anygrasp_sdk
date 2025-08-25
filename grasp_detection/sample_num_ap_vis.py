import numpy as np
import os
import matplotlib.pyplot as plt

model_list = [
    'anygrasp_raw',
    'anygrasp_ours_l1_grad_restored',
    'anygrasp_ours_l1_grad_restored_conf_0.1',
    'anygrasp_ours_l1_grad_restored_conf_0.3',
    'anygrasp_ours_l1_grad_restored_conf_0.5',
    'anygrasp_d3roma_rgbd',
    'anygrasp_d3roma_stereo'
]

camera_type = 'realsense'
sample_num_list = [5000, 8000, 15000]
experiment_root_base = '/media/2TB/result/grasp/graspnet_trans'

mean_ap_dict = {model: [] for model in model_list}
cf_rate_dict = {model: [] for model in model_list}   # 保存 collision-free rate

for sample_num in sample_num_list:
    experiment_root = os.path.join(experiment_root_base, str(sample_num))
    for model in model_list:
        root = os.path.join(experiment_root, model)
        split_ap, split_cf = [], []
        for split in ['seen', 'similar', 'novel']:
            npy_path = os.path.join(root, f'ap_test_{split}_{camera_type}_cf.npy')
            npy_path = npy_path if os.path.exists(npy_path) else npy_path.replace('_cf.npy', '.npy')
            if not os.path.exists(npy_path):
                print(f"Warning: {npy_path} not found!")
                split_ap.append(np.nan)
                split_cf.append(np.nan)
                continue

            res = np.load(npy_path)

            if res.ndim == 4:
                ap_top50 = np.mean(res[:, :, :50, :]) * 100.0
                cf_top50 = np.nan  # 原始版本没有 collision-free channel
            elif res.ndim == 5:
                ap_top50 = np.mean(res[:, :, :50, :, 0]) * 100.0
                cf_top50 = np.mean(res[:, :, :50, :, 1]) * 100.0   # collision-free rate

            split_ap.append(ap_top50)
            split_cf.append(cf_top50)

        mean_ap = np.nanmean(split_ap)
        mean_cf = np.nanmean(split_cf)

        mean_ap_dict[model].append(mean_ap)
        cf_rate_dict[model].append(mean_cf)

        print(f"Model: {model}, Samples: {sample_num}, mean AP: {mean_ap:.4f}, CF rate: {mean_cf:.4f}")

# -------- 可视化 mean AP 折线图 --------
plt.figure(figsize=(8, 6))
for model, ap_list in mean_ap_dict.items():
    plt.plot(sample_num_list, ap_list, marker='o', label=model)

plt.xlabel("Sampling Number")
plt.ylabel("mean AP (Top 50, avg over seen/similar/novel)")
plt.title("Mean AP of Different Methods vs. Sampling Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mean_ap_vs_sample_num.png', dpi=300)

# -------- 可视化 collision-free rate bar 图 --------
x = np.arange(len(sample_num_list))
bar_width = 0.1

plt.figure(figsize=(10, 6))
for i, (model, cf_list) in enumerate(cf_rate_dict.items()):
    plt.bar(x + i*bar_width, cf_list, width=bar_width, label=model)

plt.xlabel("Sampling Number")
plt.ylabel("Collision-Free Rate (%) (Top 50)")
plt.title("Collision-Free Rate of Different Methods")
plt.xticks(x + bar_width*(len(model_list)//2), sample_num_list)
plt.legend()
plt.tight_layout()
plt.savefig('collision_free_rate_bar.png', dpi=300)
import numpy as np
import os
import matplotlib.pyplot as plt

model_list = [
    # 'anygrasp_gt',
    'anygrasp_raw',
    'anygrasp_ours_l1_grad_restored',
    'anygrasp_ours_l1_grad_restored_conf_0.5',
    # 'anygrasp_drnet',
    'anygrasp_d3roma_rgbd',
    'anygrasp_d3roma_stereo'
]

camera_type = 'realsense'
sample_num_list = [5000, 8000, 15000]
experiment_root_base = '/media/2TB/result/grasp/graspnet_trans'

mean_ap_dict = {model: [] for model in model_list}

for sample_num in sample_num_list:
    experiment_root = os.path.join(experiment_root_base, str(sample_num))
    for model in model_list:
        root = os.path.join(experiment_root, model)
        split_ap = []
        for split in ['seen', 'similar', 'novel']:
            npy_path = os.path.join(root, f'ap_test_{split}_{camera_type}_cf.npy')
            npy_path = npy_path if os.path.exists(npy_path) else npy_path.replace('_cf.npy', '.npy')
            if not os.path.exists(npy_path):
                print(f"Warning: {npy_path} not found!")
                split_ap.append(np.nan)
                continue
            res = np.load(npy_path)
            if res.ndim == 4:
                ap_top50 = np.mean(res[:, :, :50, :]) * 100.0
            elif res.ndim == 5:
                ap_top50 = np.mean(res[:, :, :50, :, 0]) * 100.0
            split_ap.append(ap_top50)
        mean_ap = np.mean(split_ap)
        mean_ap_dict[model].append(mean_ap)
        print(f"Model: {model}, Samples: {sample_num}, mean AP: {mean_ap:.4f}")

# 可视化
plt.figure(figsize=(8, 6))
for model, ap_list in mean_ap_dict.items():
    plt.plot(sample_num_list, ap_list, marker='o', label=model)

plt.xlabel("Sampling Number")
plt.ylabel("mean AP (Top 50, avg over seen/similar/novel)")
plt.title("Mean AP of Different Methods vs. Sampling Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('mean_ap_vs_sample_num.png', dpi=300)

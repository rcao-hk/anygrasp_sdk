import numpy as np
import os
import pandas as pd

# method = 'ignet_v0.8.1'
# epoch_list = ['40', '45', '50', '55', '60']
# model_list = [method + '_' + i for i in epoch_list]
# model_list = [method]
experiment_root = '/media/2TB/result/grasp/graspnet_trans_full/15000'
# experiment_root = '/media/gpuadmin/rcao/result/ignet/experiment'

# model_list = ['anygrasp_gt', 'anygrasp_raw', 'anygrasp_ours_l1_grad_restored', 'anygrasp_ours_l1_grad_restored_conf', 'anygrasp_ours_l1_grad_restored_conf_0.3', 'anygrasp_drnet', 'anygrasp_d3roma_rgbd', 'anygrasp_d3roma_stereo']
model_list = ['anygrasp_ours_l1_grad_restored', 'anygrasp_ours_l1_grad_restored_conf_0.5']

topk = 1

column = ['AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP_mean']
camera_type = 'realsense'
epoch_data = []
for model in model_list:
    root = os.path.join(experiment_root, model)
    data = []
    split_ap = []
    split_cf_rate = []
    for split in ['seen', 'similar', 'novel']:
        result_path = os.path.join(root, f'ap_test_{split}_{camera_type}_cf.npy')
        result_path = result_path if os.path.exists(result_path) else result_path.replace('_cf.npy', '.npy')
        res = np.load(result_path)

        if res.ndim == 4:
            ap = res[:, :, :50, :]
        elif res.ndim == 5:
            ap = res[:, :, :50, :, 0]
            
        ap_top50 = np.mean(ap)
        print('\nEvaluation Result of Top 50 Grasps:\n----------\n{}, AP {}={:6f}'.format(camera_type, split, ap_top50))

        ap_top50_0dot2 = np.mean(ap[..., 0])
        print('----------\n{}, AP0.2 {}={:6f}'.format(camera_type, split, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(ap[..., 1])
        print('----------\n{}, AP0.4 {}={:6f}'.format(camera_type, split, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(ap[..., 2])
        print('----------\n{}, AP0.6 {}={:6f}'.format(camera_type, split, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(ap[..., 3])
        print('----------\n{}, AP0.8 {}={:6f}'.format(camera_type, split, ap_top50_0dot8))

        if res.ndim == 5:
            cf_rate = np.mean(res[:, :, 0, :, 1])
        else:
            cf_rate = 0.0
        split_ap.append(ap_top50)
        split_cf_rate.append(cf_rate)
        data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4])

    data.extend([np.mean(split_ap)])
    data.extend(split_cf_rate)
    epoch_data.append(data)
    
# data_table = pd.DataFrame(columns=column, index=model_list, data=epoch_data)
# data_table.to_csv('epoch_experiment.csv')
for model_name, data in zip(model_list, epoch_data):
    print(model_name, data)
    print("\t")

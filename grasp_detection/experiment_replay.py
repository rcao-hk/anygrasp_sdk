import numpy as np
import os
import pandas as pd

# method = 'ignet_v0.8.1'
# epoch_list = ['40', '45', '50', '55', '60']
# model_list = [method + '_' + i for i in epoch_list]
# model_list = [method]
experiment_root = '/media/2TB/result/grasp/cd'
# experiment_root = '/media/gpuadmin/rcao/result/ignet/experiment'

# model_list = ['anygrasp_gt', 'anygrasp_raw', 'anygrasp_ours_restored', 'anygrasp_ours_restored_conf', 'anygrasp_d3roma_rgbd', 'anygrasp_d3roma_stereo']
model_list = ['anygrasp_ours_l1_grad_restored', 'anygrasp_ours_l1_grad_restored_conf_0.1', 'anygrasp_ours_l1_grad_restored_conf_0.3', 'anygrasp_ours_l1_grad_restored_conf_0.5', 'anygrasp_ours_l1_grad_restored_conf_0.7', 'anygrasp_ours_l1_grad_restored_conf_0.9']
topk = 1

column = ['AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP_mean']
camera_type = 'realsense'
epoch_data = []
for model in model_list:
    root = os.path.join(experiment_root, model)
    data = []
    split_ap = []
    for split in ['seen', 'similar', 'novel']:
        res = np.load(os.path.join(root, 'ap_test_{}_{}.npy'.format(split, camera_type)))

        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Grasps:\n----------\n{}, AP {}={:6f}'.format(camera_type, split, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 {}={:6f}'.format(camera_type, split, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 {}={:6f}'.format(camera_type, split, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2])
        print('----------\n{}, AP0.6 {}={:6f}'.format(camera_type, split, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 {}={:6f}'.format(camera_type, split, ap_top50_0dot8))

        ap_top1 = np.mean(res[:, :, :topk, :])
        print('----------\n{}, TOP{} AP {}={:6f}'.format(camera_type, topk, split, ap_top1))

        ap_top1_0dot4 = np.mean(res[..., :topk, 1])
        print('----------\n{}, TOP{} AP0.4 {}={:6f}'.format(camera_type, topk, split, ap_top1_0dot4))
        
        ap_top1_0dot8 = np.mean(res[..., :topk, 3])
        print('----------\n{}, TOP{} AP0.8 {}={:6f}'.format(camera_type, topk, split, ap_top1_0dot8))
        
        split_ap.append(ap_top50)
        # data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4, ap_top1, ap_top1_0dot8, ap_top1_0dot4])
        data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4])

    data.extend([np.mean(split_ap)])
    epoch_data.append(data)
    
# data_table = pd.DataFrame(columns=column, index=model_list, data=epoch_data)
# data_table.to_csv('epoch_experiment.csv')
for model_name, data in zip(model_list, epoch_data):
    print(model_name, data)
    print("\t")

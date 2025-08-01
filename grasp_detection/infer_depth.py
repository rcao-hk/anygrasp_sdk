import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import cv2
import time
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
import open3d as o3d
import MinkowskiEngine as ME

from graspnetAPI import GraspGroup
from gsnet import AnyGrasp

from graspnetAPI.utils.utils import create_point_cloud_from_depth_image, CameraInfo

from torchvision import transforms
from noise_utils import get_workspace_mask, add_gaussian_noise_point_cloud, apply_smoothing, random_point_dropout, transform_point_cloud
from collision_detection_utils import ModelFreeCollisionDetectorTorch

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)


def get_workspace_limitation(cloud, seg, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    h, w, _ = cloud.shape
    cloud = cloud.reshape([h*w, 3])
    seg = seg.reshape(h*w)
    fg_cloud = cloud[seg>0]
    # xmin, ymin, zmin = foreground.min(axis=0)
    # xmax, ymax, zmax = foreground.max(axis=0)
    foreground = o3d.geometry.PointCloud()
    foreground.points = o3d.utility.Vector3dVector(fg_cloud)
    foreground_bb = foreground.get_axis_aligned_bounding_box()
    xmin, ymin, zmin = foreground_bb.get_min_bound()
    xmax, ymax, zmax = foreground_bb.get_max_bound()
    lims = [xmin-outlier, xmax+outlier, ymin-outlier, ymax+outlier, zmin-outlier, zmax+outlier]
    return lims

def random_sampling(points_len, sample_num):
    if points_len >= sample_num:
        idxs = np.random.choice(points_len, sample_num, replace=False)
    else:
        idxs1 = np.arange(points_len)
        idxs2 = np.random.choice(points_len, sample_num - points_len, replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return idxs


def uncertainty_guided_sampling_multimodal(uncertainty_map, sample_num, low_conf_threshold=0.5):
    """
    使用torch.multinomial进行不确定性引导的采样。
    
    参数:
    uncertainty_map (np.ndarray): 图像尺寸的不确定性图，形状为 (H, W)。
    sample_num (int): 需要采样的深度点的数量。

    返回:
    sampled_indices (torch.Tensor): 采样的像素索引。
    """
    
    # 将不确定性图转换为Tensor并归一化
    uncertainty_map = torch.tensor(uncertainty_map, dtype=torch.float32)
    
    # 归一化不确定性图（值在0和1之间）
    uncertainty_map = uncertainty_map - uncertainty_map.min() / (uncertainty_map.max() - uncertainty_map.min())
    
    # 通过不确定性图生成概率分布
    prob_distribution = 1 - uncertainty_map.flatten()
    prob_distribution[prob_distribution < low_conf_threshold] = 0.0  # 设置低置信度阈值
    
    # 采样的数量为sample_num，从概率分布中进行无放回采样
    sampled_indices = torch.multinomial(prob_distribution, sample_num, replacement=False)

    return sampled_indices


def uncertainty_guided_sampling_topk(uncertainty, sample_num):
    """
    Uncertainty-guided sampling based on depth uncertainty.
    """
    # Flatten the depth and uncertainty arrays
    uncertainty = uncertainty.flatten()

    # Get indices of the top uncertain points
    indices = np.argsort(uncertainty)[:sample_num]
    return indices


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--checkpoint_path', default='/home/smartgrasping/rcao/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar', help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--dataset_root', default='/media/2TB/dataset/graspnet', help='Where dataset is')
parser.add_argument('--sim_dataset_root', default='/media/2TB/dataset/graspnet_sim/graspnet_trans', help='Where dataset is')
parser.add_argument('--depth_result_root', default='/media/2TB/result/depth/graspnet_trans/d3roma_rgb+raw', help='Where depth results are')
parser.add_argument('--depth_type', default='restored', help='Type of depth results [gt | raw | restored | restored_conf]')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for restored depth')
parser.add_argument('--dump_dir', default='/media/2TB/result/grasp/cd/anygrasp_d3roma_rgbd', help='Dump directory [default: dump]')
parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers in point clouds')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--anno_sample_ratio', type=float, default=0.1, help='Image sample ratio for evaluation')
parser.add_argument('--scene_points_num', type=int, default=15000, help='Number of points in each scene [default: 100000]')
parser.add_argument('--apply_object_mask', action='store_true', help='Apply object mask to point cloud')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

print(cfgs)

img_width = 720
img_length = 1280
anno_sample_ratio = 0.1 # sample every 10th annotation
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
zmin, zmax = 0.0, 1.0
lims = [xmin, xmax, ymin, ymax, zmin, zmax]

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
sim_dataset_root = cfgs.sim_dataset_root
dump_dir = cfgs.dump_dir
os.makedirs(dump_dir, exist_ok=True)
vis_dir = 'vis'
os.makedirs(vis_dir, exist_ok=True)

import matplotlib
cmap = matplotlib.colormaps.get_cmap('viridis')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()

def inference(scene_idx):
    for anno_idx in range(0, 256, int(1/cfgs.anno_sample_ratio)):
        rgb_path = os.path.join(sim_dataset_root, '{:05d}/{:04d}_color.png'.format(scene_idx, anno_idx))
        if cfgs.depth_type == 'gt':
            depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
        elif cfgs.depth_type == 'raw':
            depth_path = os.path.join(sim_dataset_root, '{:05d}/{:04d}_depth_sim.png'.format(scene_idx, anno_idx))
        elif cfgs.depth_type == 'restored':
            depth_path = os.path.join(cfgs.depth_result_root, '{:05d}/{:06d}_depth.png'.format(scene_idx, anno_idx))
        elif cfgs.depth_type == 'restored_conf':
            depth_path = os.path.join(cfgs.depth_result_root, '{:05d}/{:06d}_depth.png'.format(scene_idx, anno_idx))
            conf_path = os.path.join(cfgs.depth_result_root, '{:05d}/{:06d}_conf.npy'.format(scene_idx, anno_idx))
            conf_map = np.load(conf_path)
        mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))
        meta = scio.loadmat(meta_path)

        if depth.shape[0] != img_length or depth.shape[1] != img_width:
            depth = cv2.resize(depth, (img_length, img_width), interpolation=cv2.INTER_NEAREST)
            
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)

        # if cfgs.depth_type == 'restored_conf':
        #     conf_map_vis = conf_map.copy()
        #     conf_map_vis = (conf_map_vis - conf_map_vis.min()) / (conf_map_vis.max() - conf_map_vis.min())
        #     conf_map_vis = cmap(conf_map_vis)[:, :, :3]
        #     conf_map_vis = cv2.cvtColor((conf_map_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(vis_dir, 'conf_{:05d}_{:04d}.png'.format(scene_idx, anno_idx)), conf_map_vis)
        
        depth_mask = depth > 0
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
        camera_poses = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        # lims = get_workspace_limitation(cloud, seg, outlier=0.02)
        
        mask = (depth_mask & workspace_mask)
        points = cloud[mask].astype(np.float32)
        colors = color[mask].astype(np.float32)
    
        if cfgs.depth_type == 'restored_conf':
            conf = conf_map[mask].astype(np.float32)
            scene_sample_idx = uncertainty_guided_sampling_multimodal(conf, cfgs.scene_points_num, low_conf_threshold=cfgs.conf_threshold)
            # scene_sample_idx = uncertainty_guided_sampling_topk(conf, cfgs.scene_points_num)
        else:
            scene_sample_idx = random_sampling(len(points), cfgs.scene_points_num)
        
        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(points[scene_sample_idx])
        # scene.colors = o3d.utility.Vector3dVector(colors[scene_sample_idx])
        # o3d.io.write_point_cloud(os.path.join(vis_dir, '{}_{:04d}_{:04d}.ply'.format(cfgs.depth_type, scene_idx, anno_idx)), scene)
        
        # if cfgs.depth_type == 'restored_conf':
        #     conf_vis = conf[scene_sample_idx]
        #     conf_vis = (conf_vis - conf_vis.min()) / (conf_vis.max() - conf_vis.min())
        #     conf_vis = cmap(conf_vis)[:, :3]
        #     scene_conf = o3d.geometry.PointCloud()
        #     scene_conf.points = o3d.utility.Vector3dVector(points[scene_sample_idx])
        #     scene_conf.colors = o3d.utility.Vector3dVector(conf_vis)
        #     # scene_conf.colors = o3d.utility.Vector3dVector(colors[scene_sample_idx])
        #     o3d.io.write_point_cloud(os.path.join(vis_dir, '{}_{:04d}_{:04d}_{:.2f}_conf.ply'.format(cfgs.depth_type, scene_idx, anno_idx, cfgs.conf_threshold)), scene_conf)

        sampled_points = points[scene_sample_idx] 
        sampled_colors = colors[scene_sample_idx]
            
        with torch.no_grad():
            gg, _ = anygrasp.get_grasp(sampled_points, sampled_colors, lims=lims, apply_object_mask=cfgs.apply_object_mask, dense_grasp=False, collision_detection=False)

        save_dir = os.path.join(dump_dir, 'scene_%04d'%scene_idx, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)
        
        if gg is None or len(gg) == 0:
            print('No grasps found for scene {}, annotation {}'.format(scene_idx, anno_idx))
            gg = GraspGroup()
            save_path = os.path.join(save_dir, '%04d'%anno_idx+'.npy')
            gg.save_npy(save_path)
            continue
        
        if cfgs.collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print('Inference Time:', elapsed_time)
        # elapsed_time_list.append(elapsed_time)
            
        # save grasps
        save_path = os.path.join(save_dir, '%04d'%anno_idx+'.npy')
        gg.save_npy(save_path)
        print('Saving {}, {}'.format(scene_idx, anno_idx))


scene_list = []
if split == 'test':
    for i in range(100, 190):
        scene_list.append(i)
elif split == 'test_seen':
    for i in range(100, 130):
        scene_list.append(i)
elif split == 'test_similar':
    for i in range(130, 160):
        scene_list.append(i)
elif split == 'test_novel':
    for i in range(160, 190):
        scene_list.append(i)
else:
    print('invalid split')

for scene_idx in scene_list:
    inference(scene_idx)
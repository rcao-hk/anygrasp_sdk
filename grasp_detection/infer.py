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

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test', help='Dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--checkpoint_path', default='log/checkpoint_detection.tar', help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet', help='Where dataset is')
parser.add_argument('--dump_dir', default='anygrasp', help='Dump directory [default: dump]')
parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers in point clouds')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

print(cfgs)

img_width = 720
img_length = 1280

xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
zmin, zmax = 0.0, 1.0
lims = [xmin, xmax, ymin, ymax, zmin, zmax]

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
dump_dir = cfgs.dump_dir
os.makedirs(dump_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()

def inference(scene_idx):
    for anno_idx in range(256):
        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))   
        mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))

        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)

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

        with torch.no_grad():
            gg, _ = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
                      
        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print('Inference Time:', elapsed_time)
        # elapsed_time_list.append(elapsed_time)

        # if gg is None:
        #     print('No Grasp detected after collision detection!')
        #     scene = o3d.geometry.PointCloud()
        #     scene.points = o3d.utility.Vector3dVector(points)
        #     scene.colors = o3d.utility.Vector3dVector(colors)
        #     scene = scene.voxel_down_sample(voxel_size=0.01)
        #     scene_vis = scene
        #     o3d.io.write_point_cloud('{:04d}_{:04d}.ply'.format(scene_idx, anno_idx), scene_vis)
            
        # if cfgs.debug:
        #     gg = gg.nms().sort_by_score()
        #     gg_pick = gg[0:50]

        #     scene = o3d.geometry.PointCloud()
        #     scene.points = o3d.utility.Vector3dVector(points)
        #     scene.colors = o3d.utility.Vector3dVector(colors)
        #     scene = scene.voxel_down_sample(voxel_size=0.01)
            
        #     trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        #     scene.transform(trans_mat)
        #     grippers = gg_pick.to_open3d_geometry_list()
        #     grippers_pcd = []
        #     for gripper in grippers:
        #         gripper.transform(trans_mat)
        #         grippers_pcd.append(gripper.sample_points_uniformly(number_of_points=200))
        #     scene_vis = scene
        #     for gripper_pcd in grippers_pcd:
        #         scene_vis = scene_vis + gripper_pcd
        #     # scene_vis = inst_vis + proj_scene + axis_pcd
        #     o3d.io.write_point_cloud('{:04d}_{:04d}.ply'.format(scene_idx, anno_idx), scene_vis)
            
        # save grasps
        save_dir = os.path.join(dump_dir, 'scene_%04d'%scene_idx, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)
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
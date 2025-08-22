import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import copy
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.io as scio
import open3d as o3d
from graspnetAPI.utils.utils import create_point_cloud_from_depth_image, CameraInfo, transform_points, generate_scene_model
from PIL import Image
from graspnetAPI import GraspGroup
import cv2

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
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
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_points(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

dataset_root = '/media/2TB/dataset/graspnet_sim/graspnet_trans_full'
raw_dataset_root = '/media/2TB/dataset/graspnet'

vis_root = os.path.join('vis', 'grasp_vis')
os.makedirs(vis_root, exist_ok=True)

restored_depth_root = '/media/2TB/result/depth/graspnet_trans_full/dreds_dav2_complete_obs_iter_unc_cali_convgru_l1_only_0.5_l1+grad_sigma_conf_320x180/vitl'
camera_type = 'realsense'

grasp_data_root = '/media/2TB/result/grasp/graspnet_trans_full/15000'
method1 = 'gsnet_virtual_ours_restored'
method2 = 'gsnet_virtual_ours_restored_conf_0.5'

width = 1280
height = 720
sample_ratio = 1
conf_thres = 0.5
sample_num = 15000
scene_list = range(102, 103)

import pickle
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import create_table_points, voxel_sample_points, eval_grasp, get_scene_name, parse_posevector, load_dexnet_model
from graspnetAPI.utils.xmlhandler import xmlReader
import torch

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
    uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min())
    
    # 通过不确定性图生成概率分布
    prob_distribution = 1 - uncertainty_map.flatten()
    prob_distribution[prob_distribution < low_conf_threshold] = 0.0  # 设置低置信度阈值
    
    # 采样的数量为sample_num，从概率分布中进行无放回采样
    if len(prob_distribution) < sample_num:
        print("Sample number exceeds the number of available pixels.")
        sampled_indices = torch.multinomial(prob_distribution, sample_num, replacement=True)
    else:
        sampled_indices = torch.multinomial(prob_distribution, sample_num, replacement=False)

    return sampled_indices


def get_scene_models(scene_id, ann_id):
    '''
        return models in model coordinate
    '''
    model_dir = os.path.join(raw_dataset_root, 'models')
    # print('Scene {}, {}'.format(scene_id, camera))
    scene_reader = xmlReader(os.path.join(raw_dataset_root, 'scenes', get_scene_name(scene_id), camera_type, 'annotations', '%04d.xml' % (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    model_list = []
    dexmodel_list = []
    for posevector in posevectors:
        obj_idx, _ = parse_posevector(posevector)
        obj_list.append(obj_idx)
    for obj_idx in obj_list:
        model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
        dex_cache_path = os.path.join(raw_dataset_root, 'dex_models', '%03d.pkl' % obj_idx)
        if os.path.exists(dex_cache_path):
            with open(dex_cache_path, 'rb') as f:
                dexmodel = pickle.load(f)
        else:
            dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
        points = np.array(model.points)
        model_list.append(points)
        dexmodel_list.append(dexmodel)
    return model_list, dexmodel_list, obj_list


def get_model_poses(scene_id, ann_id):
    '''
    **Input:**

    - scene_id: int of the scen index.

    - ann_id: int of the annotation index.

    **Output:**

    - obj_list: list of int of object index.

    - pose_list: list of 4x4 matrices of object poses.

    - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

    - align mat: 4x4 matrix of camera relative to the table.
    '''
    scene_dir = os.path.join(raw_dataset_root, 'scenes')
    camera_poses_path = os.path.join(raw_dataset_root, 'scenes', get_scene_name(scene_id), camera_type, 'camera_poses.npy')
    camera_poses = np.load(camera_poses_path)
    camera_pose = camera_poses[ann_id]
    align_mat_path = os.path.join(raw_dataset_root, 'scenes', get_scene_name(scene_id), camera_type, 'cam0_wrt_table.npy')
    align_mat = np.load(align_mat_path)
    # print('Scene {}, {}'.format(scene_id, camera))
    scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), camera_type, 'annotations', '%04d.xml'% (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)
    return obj_list, pose_list, camera_pose, align_mat
    
def eval_scene_grasp(scene, scene_id, ann_id, dump_folder, max_width=0.1, TOP_K=50):
    config = get_config()
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    
    list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

    model_list, dexmodel_list, _ = get_scene_models(scene_id, ann_id=0)

    model_sampled_list = list()
    for model in model_list:
        model_sampled = voxel_sample_points(model, 0.008)
        model_sampled_list.append(model_sampled)

    scene_accuracy = []
    grasp_list_list = []
    score_list_list = []
    collision_list_list = []

    # for ann_id in range(256):
    grasp_group = GraspGroup().from_npy(os.path.join(dump_folder, get_scene_name(scene_id), camera_type, '%04d.npy' % (ann_id,)))
    _, pose_list, camera_pose, align_mat = get_model_poses(scene_id, ann_id)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

    # clip width to [0,max_width]
    gg_array = grasp_group.grasp_group_array
    min_width_mask = (gg_array[:,1] < 0)
    max_width_mask = (gg_array[:,1] > max_width)
    gg_array[min_width_mask,1] = 0
    gg_array[max_width_mask,1] = max_width
    grasp_group.grasp_group_array = gg_array

    grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008, TOP_K = TOP_K)

    # remove empty
    grasp_list = [x for x in grasp_list if len(x) != 0]
    score_list = [x for x in score_list if len(x) != 0]
    collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

    if len(grasp_list) == 0:
        grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
        scene_accuracy.append(grasp_accuracy)
        grasp_list_list.append([])
        score_list_list.append([])
        collision_list_list.append([])
        print('Mean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id),np.mean(grasp_accuracy[:,:]))

    # concat into scene level
    grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)

    # === D) 统计碰撞/非碰撞数量 + 分数分布
    print(f"[stats] total grasps={len(score_list)}, collision-free={np.sum(~collision_mask_list)}, collided={np.sum(collision_mask_list)}")
    print(f"[stats] score min/mean/median/max={np.min(score_list):.4f}/{np.mean(score_list):.4f}/{np.median(score_list):.4f}/{np.max(score_list):.4f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(score_list[~collision_mask_list], bins=30, alpha=0.7, label='non-collide')
    plt.hist(score_list[collision_mask_list], bins=30, alpha=0.7, label='collide')
    plt.legend(); plt.title('score histogram'); plt.savefig(os.path.join(vis_root, f'scene_{scene_id:04d}_ann_{ann_id:04d}_score_hist.png'), dpi=300)

    gg = GraspGroup(copy.deepcopy(grasp_list))
    scores = np.array(score_list)
    scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
    scores[collision_mask_list] = 0.3
    gg.scores = scores
    gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float32)
    grasps_geometry = gg.to_open3d_geometry_list()
    for grasps in grasps_geometry:
        scene += grasps.sample_points_uniformly(number_of_points=1000)
    method_name = os.path.basename(dump_folder)
    o3d.io.write_point_cloud(os.path.join(vis_root, f'scene_{scene_id:04d}_ann_{ann_id:04d}_{method_name}_grasp.ply'), scene)
            
    # sort in scene level
    grasp_confidence = grasp_list[:,0]
    indices = np.argsort(-grasp_confidence)
    grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

    grasp_list_list.append(grasp_list)
    score_list_list.append(score_list)
    collision_list_list.append(collision_mask_list)

    #calculate AP
    grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
    for fric_idx, fric in enumerate(list_coe_of_friction):
        for k in range(0,TOP_K):
            if k+1 > len(score_list):
                grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
            else:
                grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

    grasp_accuracy_v2 = np.zeros((TOP_K,len(list_coe_of_friction)))
    valid_scores = score_list[~collision_mask_list]
    for j, fric in enumerate(list_coe_of_friction):
        for k in range(TOP_K):
            top = min(k+1, len(valid_scores))
            if top == 0:
                grasp_accuracy_v2[k,j] = 0
            else:
                pos = np.sum(((valid_scores[:top] <= fric) & (valid_scores[:top] > 0)).astype(int))
                grasp_accuracy_v2[k,j] = pos / top  # 用真实可用提案数

    print(f"Mean AP (v1/orig) = {np.mean(grasp_accuracy):.4f}")
    print(f"Mean AP (v2/no-collision, real denom) = {np.mean(grasp_accuracy_v2):.4f}")
    
    # print('Mean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:,:]))

def get_vis_colormap(depth, mask, min_value=0.0, max_value=2.0, colormap='viridis'):
    """
    Get a colormap for visualizing depth data.

    Args:
        depth (numpy.ndarray): Depth data to be visualized.

    Returns:
        numpy.ndarray: Colormap applied to the depth data.
    """
    # Normalize the depth values to the range [0, 1]
    # depth[~mask] = 0
    depth_normalized = np.zeros_like(depth)
    # depth_normalized[mask] = (depth[mask] - np.min(depth[mask])) / (np.max(depth[mask]) - np.min(depth[mask]))
    depth_normalized[mask] = (depth[mask] - min_value) / (max_value - min_value)
    # depth_normalized[~mask] = 0
    # Apply a colormap (e.g., 'viridis') to the normalized depth values
    if colormap == 'viridis':
        colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    elif colormap == 'jet':
        colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colormap[~mask] = [0, 0, 0]  # Set invalid pixels to black
    return colormap



for scene_idx in tqdm(scene_list):
    for anno_idx in range(108, 109, int(1/sample_ratio)):

        rgb_path = os.path.join(dataset_root,
                                '{:05d}/{:04d}_color.png'.format(scene_idx, anno_idx))
        depth_path = os.path.join(dataset_root,
                                  '{:05d}/{:04d}_depth_sim.png'.format(scene_idx, anno_idx))

        restored_depth_path = os.path.join(restored_depth_root, '{:05d}/{:06d}_depth.png'.format(scene_idx, anno_idx))
        
        restored_depth_conf_path = os.path.join(restored_depth_root, '{:05d}/{:06d}_conf.npz'.format(scene_idx, anno_idx))
        meta_path = os.path.join(raw_dataset_root,
                                 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera_type, anno_idx))
        
        mask_path = os.path.join(raw_dataset_root,
                                 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera_type, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        restored_depth = np.array(Image.open(restored_depth_path))
        restored_depth_conf = np.load(restored_depth_conf_path)['conf']
        seg = np.array(Image.open(mask_path))
        
        meta = scio.loadmat(meta_path)
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)

        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
        restored_cloud = create_point_cloud_from_depth_image(restored_depth, camera_info, organized=True)
        
        depth_mask = (depth > 0)
        camera_poses = np.load(
            os.path.join(raw_dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera_type)))
        align_mat = np.load(
            os.path.join(raw_dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera_type)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cv2.imwrite(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:04d}_color.png'), (color[:, :,::-1] * 255).astype(np.uint8))
        # restored_depth_conf_norm = (restored_depth_conf - restored_depth_conf.min()) / (restored_depth_conf.max() - restored_depth_conf.min())
        vaild_mask = (restored_depth > 0)
        restored_depth_conf = np.clip(restored_depth_conf, 0.0, 1.0)
        restored_depth_conf_colormap = get_vis_colormap(restored_depth_conf, vaild_mask, 0.0, 1.0, colormap='viridis')
        
        cv2.imwrite(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:04d}_conf.png'), (restored_depth_conf_colormap).astype(np.uint8))
        
        
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        restored_cloud_masked = restored_cloud[mask]
        restored_depth_conf_masked = restored_depth_conf[mask]
        seg_masked = seg[mask]

        choose_idx = random_sampling(len(restored_cloud_masked), sample_num)
        restored_conf_choose_idx = uncertainty_guided_sampling_multimodal(restored_depth_conf_masked, sample_num, conf_thres)
        
        restored_conf_cloud_masked = restored_cloud_masked[restored_conf_choose_idx]
        restored_conf_color_masked = color_masked[restored_conf_choose_idx]

        restored_cloud_masked = restored_cloud_masked[choose_idx]
        restored_color_masked = color_masked[choose_idx]
        
        restored_scene = o3d.geometry.PointCloud()
        restored_scene.points = o3d.utility.Vector3dVector(restored_cloud_masked)
        restored_scene.colors = o3d.utility.Vector3dVector(restored_color_masked)
        
        restored_conf_scene = o3d.geometry.PointCloud()
        restored_conf_scene.points = o3d.utility.Vector3dVector(restored_conf_cloud_masked)
        restored_conf_scene.colors = o3d.utility.Vector3dVector(restored_conf_color_masked)
        
        # restored_scene_vis = scene.voxel_down_sample(voxel_size=0.002)
        # restored_scene_vis = scene
        # restored_conf_scene_vis = copy.deepcopy(restored_scene_vis)

        # o3d.io.write_point_cloud(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:03d}_restored.ply'), restored_scene)
        # o3d.io.write_point_cloud(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:03d}_restored_conf.ply'), restored_conf_scene)
        
        # o3d.io.write_point_cloud(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:03d}_downsampled.ply'), downsampled_scene)
        
        restored_grasp = GraspGroup()
        restored_grasp.from_npy(os.path.join(grasp_data_root, method1, 'scene_{:04d}/{}/{:04d}.npy'.format(scene_idx, camera_type, anno_idx)))

        restored_grasp = restored_grasp.sort_by_score()
        restored_grasp = restored_grasp.nms()
        restored_grasp = restored_grasp[:50]
        restored_grasp_vis = restored_grasp.to_open3d_geometry_list()

        # for grasp in restored_grasp_vis:
        #     restored_scene_vis += grasp.sample_points_uniformly(number_of_points=1000)
            
        restored_grasp_conf = GraspGroup()
        restored_grasp_conf.from_npy(os.path.join(grasp_data_root, method2, 'scene_{:04d}/{}/{:04d}.npy'.format(scene_idx, camera_type, anno_idx)))
        
        restored_grasp_conf = restored_grasp_conf.sort_by_score()
        restored_grasp_conf = restored_grasp_conf.nms()
        restored_grasp_conf = restored_grasp_conf[:50]
        restored_grasp_conf_vis = restored_grasp_conf.to_open3d_geometry_list()
        
        # for grasp in restored_grasp_conf_vis:
        #     restored_conf_scene_vis += grasp.sample_points_uniformly(number_of_points=1000)
            
        # o3d.io.write_point_cloud(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:03d}_restored.ply'), restored_scene_vis)
        # o3d.io.write_point_cloud(os.path.join(vis_root, f'scene_{scene_idx:04d}_anno_{anno_idx:03d}_restored_conf.ply'), restored_conf_scene_vis)
        
        eval_scene_grasp(restored_scene, scene_idx, anno_idx, os.path.join(grasp_data_root, method1))
        eval_scene_grasp(restored_conf_scene, scene_idx, anno_idx, os.path.join(grasp_data_root, method2))
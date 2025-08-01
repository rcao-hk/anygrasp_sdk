import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
from PIL import Image
import open3d as o3d
import cv2
import scipy.io as scio
import OpenEXR
import Imath
from tqdm import tqdm

class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr
    
    
width = 1280
height = 720

scene_idx = 100
scene_list = range(100, 190)
camera_type = 'realsense'
dataset_root = '/media/2TB/dataset/graspnet_sim/graspnet_trans'
virtual_root = '/media/2TB/dataset/graspnet/virtual_scenes'
real_dataset_root = '/media/2TB/dataset/graspnet'
vis_root = os.path.join('vis', 'graspnet_trans')
os.makedirs(vis_root, exist_ok=True)

# for anno_idx in range(256):
for scene_idx in tqdm(scene_list):
    for anno_idx in [0]:
        rgb_path = os.path.join(dataset_root, '{:05d}'.format(scene_idx), '{:04d}_color.png'.format(anno_idx))
        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        
        # virtual_depth_path = rgb_path.replace('_color.png', '_depth_0.exr')
        # # virtual_depth = cv2.imread(virtual_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # # virtual_depth = virtual_depth[:, :, 0]
        # virtual_depth = exr_loader(virtual_depth_path, ndim=1)

        # virtual_mask_path = rgb_path.replace('_color.png', '_mask.exr')
        # # virtual_mask = cv2.imread(virtual_mask_path, cv2.IMREAD_UNCHANGED)
        # # virtual_mask = cv2.imread(virtual_mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # # virtual_mask = virtual_mask[:, :, 0]
        
        # # print(virtual_mask)
        # virtual_mask = exr_loader(virtual_mask_path, ndim=1)
        
        # print("Image type:", type(virtual_mask))
        # print("Image dtype:", virtual_mask.dtype)
        
        # virtual_mask = np.array(virtual_mask * 255, dtype=np.float32)
        # virtual_mask[virtual_mask==255] = 0
        
        # print(virtual_mask.shape, np.unique(virtual_mask))
        # virtual_mask = cv2.resize(virtual_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('mask.png', virtual_mask)
        
        virtual_depth_path = os.path.join(virtual_root, 'scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera_type, anno_idx))
        virtual_depth = np.array(Image.open(virtual_depth_path))
    
        meta_path = os.path.join(real_dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera_type, anno_idx))
        # real_depth_path = os.path.join(real_dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera_type, anno_idx))
        real_depth_path = os.path.join(dataset_root, '{:05d}'.format(scene_idx), '{:04d}_depth_sim.png'.format(anno_idx))
        simsense_real_depth = np.array(Image.open(real_depth_path))
        
        # dred_real_depth_path = os.path.join(dataset_root, '{:05d}'.format(scene_idx), '{:04d}_simDepthImage.exr'.format(anno_idx))
        
        # dred_real_depth =  cv2.imread(dred_real_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # dred_real_depth = dred_real_depth[:, :, 0]
        # print("dred_real_depth shape:", dred_real_depth.shape)
        # print('min:', np.min(dred_real_depth), 'max:', np.max(dred_real_depth))
        
        # # if len(dred_real_depth.shape) == 3:
        #     
        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        poses = meta['poses']
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_fov = 2 * np.arctan(width / (2 * intrinsics[0, 0]))

        # print(intrinsics)
        # print(intrinsics[0, 0], camera_fov)
        
        # vis real depth
        real_depth_vis = simsense_real_depth / factor_depth
        real_depth_vis = real_depth_vis - np.min(real_depth_vis) / (np.max(real_depth_vis) - np.min(real_depth_vis))
        real_depth_vis = (real_depth_vis * 255).astype(np.uint8)
        real_depth_vis = cv2.applyColorMap(real_depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(vis_root, '{}_{:04d}_simsense_depth.png'.format(scene_idx, anno_idx)), real_depth_vis)
        
        # real_depth_vis = dred_real_depth
        # real_depth_vis = real_depth_vis - np.min(real_depth_vis) / (np.max(real_depth_vis) - np.min(real_depth_vis))
        # real_depth_vis = (real_depth_vis * 255).astype(np.uint8)
        # real_depth_vis = cv2.applyColorMap(real_depth_vis, cv2.COLORMAP_JET)
        # cv2.imwrite(os.path.join(vis_root, '{}_{:04d}_dreds_depth.png'.format(scene_idx, anno_idx)), real_depth_vis)
        
        color_vis = (color * 255).astype(np.uint8)
        color_vis = cv2.cvtColor(color_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(vis_root, '{}_{:04d}_color.png'.format(scene_idx, anno_idx)), color_vis)
        
        # camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        # virtual_cloud = create_point_cloud_from_depth_image(virtual_depth, camera_info, organized=True)
        # simsense_real_cloud = create_point_cloud_from_depth_image(simsense_real_depth, camera_info, organized=True)
        
        # dred_real_cloud = create_point_cloud_from_depth_image(dred_real_depth*factor_depth, camera_info, organized=True)
        
        # virtual_scene = o3d.geometry.PointCloud()
        # virtual_scene.points = o3d.utility.Vector3dVector(virtual_cloud.reshape(-1, 3))
        # virtual_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))

        # simsense_real_scene = o3d.geometry.PointCloud()
        # simsense_real_scene.points = o3d.utility.Vector3dVector(simsense_real_cloud.reshape(-1, 3))
        # # real_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        # simsense_real_scene.paint_uniform_color([1.0, 0.0, 0.0])
        # # scene = virtual_scene + real_scene
        
        # dred_real_scene = o3d.geometry.PointCloud()
        # dred_real_scene.points = o3d.utility.Vector3dVector(dred_real_cloud.reshape(-1, 3))
        # dred_real_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        # dred_real_scene.paint_uniform_color([0.0, 0.0, 1.0])
        
        # virtual_scene = virtual_scene.voxel_down_sample(voxel_size=0.002)
        # simsense_real_scene = simsense_real_scene.voxel_down_sample(voxel_size=0.002)
        # dred_real_scene = dred_real_scene.voxel_down_sample(voxel_size=0.002)
        
        # scene = virtual_scene + simsense_real_scene + dred_real_scene
        # o3d.io.write_point_cloud(os.path.join(vis_root, '{}_{:04d}_scene.ply'.format(scene_idx, anno_idx)), scene)

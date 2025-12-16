#!/usr/bin/env python3
import os
import struct
from typing import Any, Dict, Union
from collections import defaultdict
import yaml
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, QhullError

def numeric_sort_key(filename):
    """1.pcd, 2.pcd, 1001.pcd, 1002.pcd ..."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return int(base.split('.')[0]) if base.split('.')[0].isdigit() else 0

def load_from_yaml(yaml_path, field_path):
    # load_from_yaml('debug_data.yaml', 'numpy.transform_matrix')
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    if isinstance(field_path, str):
        field_path = field_path.split('.')

    data = config
    for key in field_path:
        if key not in data:
            print(f"field path {key} not found")
            return None
        data = data[key]
    
    return data

def write_to_yaml(
    path: str, 
    data: Any,
    field_path: Union[str, list],
    create_if_not_exist: bool = True
    ) -> bool:
    """
    Example usage:
    write_yaml_config("path/to/config.yaml", transform_matrix, "numpy.transform_matrix")
    """
    config = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
    if isinstance(data, np.ndarray):
        data = data.flatten().tolist()
    if isinstance(field_path, str):
        field_path = field_path.split('.')
    
    # recursive
    current = config
    for i, key in enumerate(field_path[:-1]):
        if key not in current:
            if create_if_not_exist:
                current[key] = {}
            else:
                print(f"Error: Field path {'/'.join(field_path[:i+1])} not found")
                return False
        current = current[key]
    
    # overwrite
    current[field_path[-1]] = data
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)
    return True

# do not work properly with numpy 2.2.6
# def pack_rgb_float(r, g, b):
#     """将RGB值打包为PCD格式的float32"""
#     packed_int = (r << 16) | (g << 8) | b
#     rgb_float_bytes = struct.pack('I', packed_int)
#     return struct.unpack('f', rgb_float_bytes)[0]

def pack_rgb_float(r, g, b):
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return np.array([packed], dtype=np.uint32).view(np.float32)[0]

def pack_rgb_uint32(r, g, b):
    return np.uint32((int(r) << 16) | (int(g) << 8) | int(b))

def parse_camera_matrix(cam_matrix):
    """return fx, fy, cx, cy, skew"""
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    skew = cam_matrix[0, 1]
    return fx, fy, cx, cy, skew

def apply_transform(points, transform):
    """points: Nx3, transform: 4x4"""
    points_hom = np.hstack([points, np.ones((len(points),1))])
    return (points_hom @ transform.T)[:,:3]

def batch_transform(df, transform):
    coords = df[['x','y','z']].values
    return pd.DataFrame(
        apply_transform(coords, transform),
        columns=['x','y','z'],
        index=df.index
    )

def get_inv_transform(R_awrtb: np.ndarray, t_awrtb: np.ndarray) -> np.ndarray:
    """
    R_awrtb (a w.r.t b): 3x3, used to convert coords in a to b
    t_awrtb (a w.r.t b): 3x1
    return 4x4, used to convert coords in b to a
    """
    # for precision problem, do not use transform and then inverse
    R_bwrta = R_awrtb.T
    t_bwrta = -R_bwrta @ t_awrtb
    T_bwrta = np.identity(4, dtype=np.float64)
    T_bwrta[:3, :3] = R_bwrta
    T_bwrta[:3, 3] = t_bwrta.flatten() # flatten to a 1D array for assignment

    return T_bwrta

def get_transform_matrix(para_list):
    """
    para_list: [x y z roll pitch yaw] wrt a frame
    """
    roll = para_list[3]
    pitch = para_list[4]
    yaw = para_list[5]

    R_b_wrt_a = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    t_b_wrt_a = np.array(para_list[0:3])
    T_b_wrt_a = np.eye(4)
    T_b_wrt_a[:3, :3] = R_b_wrt_a
    T_b_wrt_a[:3, 3] = t_b_wrt_a

    T_a_wrt_b = np.linalg.inv(T_b_wrt_a)
    R_a_wrt_b = T_a_wrt_b[:3, :3]
    t_a_wrt_b = T_a_wrt_b[:3, 3]
    return T_b_wrt_a, T_a_wrt_b


def get_transform_matrix_wrt_optical(T_b_wrt_camera):
    T_optical_wrt_camera = np.array([
        [ 0,  0,  1,  0],
        [-1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1]
        ])
    
    T_camera_wrt_optical = np.linalg.inv(T_optical_wrt_camera)
    T_b_wrt_optical = T_camera_wrt_optical @ T_b_wrt_camera

    # given a lidar point  P_lidar = [X, Y, Z, 1], lidar -> camlink -> optical  (right to left, _wrt_x, x is the target frame)
    # T_lidar_wrt_optical @ P_lidar = (T_camlink_wrt_optical @ T_lidar_wrt_camlink) @ P_lidar
    # need K to get image coords
    # p = K @ [R|t] @ P

    return T_b_wrt_optical

def get_transform_matrix_copy(para_list):
    """
    para_list: [x y z roll pitch yaw]
    """
    roll = para_list[3]
    pitch = para_list[4]
    yaw = para_list[5]

    R_local_wrt_base = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    t_local_wrt_base = np.array(para_list[0:3])
    T_local_wrt_base = np.eye(4)
    T_local_wrt_base[:3, :3] = R_local_wrt_base
    T_local_wrt_base[:3, 3] = t_local_wrt_base

    T_optical_wrt_camlink = np.array([
        [ 0,  0,  1,  0],
        [-1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1]
    ])
    print("--- T_optical_wrt_camlink (Camera_Optical w.r.t. Camera_Link) ---")
    print(T_optical_wrt_camlink)

    T_lidar_wrt_camlink = np.linalg.inv(T_local_wrt_base)
    T_camlink_wrt_optical = np.linalg.inv(T_optical_wrt_camlink)
    T_lidar_wrt_optical = T_camlink_wrt_optical @ T_lidar_wrt_camlink
    print(f"final matrix:\n{T_lidar_wrt_optical}")

    Rcl_final = T_lidar_wrt_optical[:3, :3]
    Pcl_final = T_lidar_wrt_optical[:3, 3]

    # given a lidar point  P_lidar = [X, Y, Z, 1], lidar -> camlink -> optical  (right to left, _wrt_x, x is the target frame)
    # T_lidar_wrt_optical @ P_lidar = (T_camlink_wrt_optical @ T_lidar_wrt_camlink) @ P_lidar
    # need K to get image coords
    # p = K @ [R|t] @ P


    print("Rcl (Rotation of LiDAR frame w.r.t. Camera Optical frame):")
    print(Rcl_final)
    print("Pcl (Translation of LiDAR frame w.r.t. Camera Optical frame):")
    print(Pcl_final)
    print("Rcl in yaml:")
    print(f"[{', '.join(f'{x:.8f}' for x in Rcl_final.ravel())}]")
    print("Pcl in yaml:")
    print(f"[{', '.join(f'{x:.8f}' for x in Pcl_final)}]")

def ransac_ground_by_grid_area(points, grid_size=5.0, num_iterations=100, distance_threshold=0.1, vertical_threshold=0.9):
    """
    Finds the ground plane by RANSAC, incorporating two key improvements:
    1. Spatial Grid Sampling: Samples points from different spatial grids to avoid bias from dense clusters.
    2. Area-based Scoring: The score of a candidate plane is weighted by the XY area covered by its inliers.
    """
    best_score = -1
    best_plane = None
    best_inlier_indices = None
    num_points = len(points)
    z_axis = np.array([0, 0, 1])

    if num_points < 3:
        return None, None

    # --- Strategy 2: Pre-computation for Grid Sampling ---
    grid_bins = np.floor(points[:, :2] / grid_size).astype(int)
    
    grid_map = defaultdict(list)
    for i, grid_coord in enumerate(grid_bins):
        grid_map[tuple(grid_coord)].append(i)
        
    unique_grids = list(grid_map.keys())
    
    if len(unique_grids) < 3:
        # Not enough spatial diversity for this method to work.
        print("Warning: Not enough unique grids (<3) for spatial sampling. Cannot find plane.")
        return None, None

    for i in range(num_iterations):
        # 1. Spatial Grid Sampling
        try:
            grid_keys_indices = np.random.choice(len(unique_grids), 3, replace=False)
            selected_grid_keys = [unique_grids[i] for i in grid_keys_indices]
            
            sample_indices = [np.random.choice(grid_map[key]) for key in selected_grid_keys]
            sample_points = points[sample_indices]
        except ValueError:
            continue

        # 2. Fit a plane to the 3 points
        p1, p2, p3 = sample_points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        
        if normal[2] < 0:
            normal = -normal
            
        a, b, c = normal
        d = -np.dot(normal, p1)
        
        # 3. Custom Validation
        if abs(np.dot(normal, z_axis)) < vertical_threshold:
            continue

        # 4. Find inliers
        distances = np.abs(np.dot(points, normal) + d)
        inlier_mask = distances < distance_threshold
        inlier_indices = np.where(inlier_mask)[0]
        num_inliers = len(inlier_indices)

        if num_inliers < 10:
            continue

        # 5. New Robust Scoring
        # 5a. Original robust part (points below plane)
        outlier_mask = ~inlier_mask
        outlier_points = points[outlier_mask]
        
        num_below = 0
        if len(outlier_points) > 0:
            signed_distances = np.dot(outlier_points, normal) + d
            num_below = np.sum(signed_distances < 0)
            
        robust_score_part = num_inliers / (num_below + 1)

        # 5b. Strategy 3: Area-based scoring part
        inlier_points_xy = points[inlier_indices, :2]
        
        area_score_part = 0.0
        if inlier_points_xy.shape[0] >= 3:
            try:
                hull = ConvexHull(inlier_points_xy)
                area_score_part = hull.volume # In 2D, .volume is the area
            except (QhullError, ValueError):
                # This can happen if all points are collinear. Area is effectively zero.
                area_score_part = 1e-4 # Assign a very small area

        # Combine scores: multiply the original score by the area.
        # Add 1.0 to area to prevent score from becoming zero for small, valid planes.
        score = robust_score_part * (area_score_part + 1.0)

        # 6. Keep track of the best model
        if score > best_score:
            best_score = score
            best_plane = np.array([a, b, c, d])
            best_inlier_indices = inlier_indices
    
    if best_plane is not None:
        print(f"Best plane found with score: {best_score:.2f}")
        print(f"Plane equation: {best_plane[0]:.3f}x + {best_plane[1]:.3f}y + {best_plane[2]:.3f}z + {best_plane[3]:.3f} = 0")
        print(f"Found {len(best_inlier_indices)} ground points.")
    else:
        print("Could not find a suitable ground plane with the new method.")
        
    return best_plane, best_inlier_indices



def determine_up_vector(points, trajectory_points, return_plane=False):
    """
    Determines the "up" vector by finding an initial ground plane and comparing it to the trajectory.
    This correctly handles both Z-up and Z-down coordinate systems.
    """
    import open3d as o3d
    # Use a large random sample for efficiency
    sample_indices = np.random.choice(len(points), min(len(points), 100000), replace=False)
    
    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(points[sample_indices])
    
    plane_model, _ = pcd_sample.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100)
    
    if plane_model is None or len(plane_model) < 4:
        if return_plane:
            return np.array([0, 0, 1]), None
        return np.array([0, 0, 1])

    plane_normal = plane_model[:3]
    plane_d = plane_model[3]
    
    # Use the average trajectory point for a stable comparison
    avg_trajectory_point = np.mean(trajectory_points, axis=0)
    
    # Calculate signed distance from the trajectory to the plane
    signed_distance = np.dot(plane_normal, avg_trajectory_point) + plane_d
    
    # The vehicle is always physically ABOVE the ground.
    if signed_distance > 0:
        up_vector = plane_normal
    else:
        up_vector = -plane_normal
        plane_model *= -1 # Invert the plane equation as well for consistency
        
    if return_plane:
        return up_vector, plane_model
    return up_vector


def get_initial_ground_seeds(points, trajectory_points, up_vector):
    """
    Finds a high-confidence set of initial ground points using a strict RANSAC approach.
    """
    import open3d as o3d
    # Use a large sample of points for a stable initial plane
    sample_indices = np.random.choice(len(points), min(len(points), 100000), replace=False)
    sample_points = points[sample_indices]

    # Use Open3D's RANSAC for simplicity and robustness
    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(sample_points)
    
    # Stricter distance threshold for high confidence
    plane_model, inlier_indices = pcd_sample.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
    
    if len(inlier_indices) < 100:
        return np.array([], dtype=int)

    # Ensure the plane normal is consistent with the determined 'up' direction
    normal = plane_model[:3]
    if np.dot(normal, up_vector) < 0:
        normal = -normal
        plane_model[:3] = normal
        plane_model[3] *= -1

    # Final check on the plane's validity
    avg_traj_z = np.mean(trajectory_points, axis=0)
    signed_dist = np.dot(normal, avg_traj_z) + plane_model[3]
    if not (1.0 < signed_dist < 3.0):
        pass # Proceeding cautiously as per log message in original script

    # Return the original indices of the inlier points
    return sample_indices[inlier_indices]


def grow_ground_region(pcd_full, up_vector, seed_indices, neighbor_radius=0.5, normal_angle_thresh=15.0, height_thresh=0.15):
    """
    Expands the ground surface from initial seeds using a region growing algorithm.
    """
    from collections import deque
    import open3d as o3d
    
    num_points = len(pcd_full.points)
    is_ground = np.zeros(num_points, dtype=bool)
    is_processed = np.zeros(num_points, dtype=bool)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)
    
    pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=neighbor_radius, max_nn=30))
    pcd_full.orient_normals_to_align_with_direction(orientation_reference=up_vector)
    all_normals = np.asarray(pcd_full.normals)
    
    q = deque(seed_indices)
    is_ground[seed_indices] = True
    is_processed[seed_indices] = True
    
    cos_angle_thresh = np.cos(np.radians(normal_angle_thresh))

    while q:
        current_idx = q.popleft()
        current_point = np.asarray(pcd_full.points)[current_idx]
        current_normal = all_normals[current_idx]

        [k, neighbor_indices, _] = pcd_tree.search_radius_vector_3d(current_point, neighbor_radius)
        
        for neighbor_idx in neighbor_indices:
            if is_processed[neighbor_idx]:
                continue
            
            is_processed[neighbor_idx] = True
            
            neighbor_point = np.asarray(pcd_full.points)[neighbor_idx]
            neighbor_normal = all_normals[neighbor_idx]
            
            if np.dot(current_normal, neighbor_normal) < cos_angle_thresh:
                continue
            
            if abs(np.dot(neighbor_point - current_point, up_vector)) > height_thresh:
                continue
            
            is_ground[neighbor_idx] = True
            q.append(neighbor_idx)
            
    return np.where(is_ground)[0]


def segment_ground_final(points_df, trajectory_df, grid_size=50.0, search_radius=30.0, step_size=5.0, max_ransac_points=20000):
    """
    Final optimized ground segmentation using spatial partitioning, RANSAC subsampling, and a height constraint.
    (This is the core logic from test_0926_final_v2.py, adapted as a helper function)
    """
    import open3d as o3d
    from collections import defaultdict

    all_points = points_df[['x', 'y', 'z']].values
    trajectory_points = trajectory_df[['x', 'y', 'z']].values
    
    if len(all_points) < 3 or len(trajectory_points) == 0:
        return np.array([], dtype=int)

    up_vector, initial_plane = determine_up_vector(all_points, trajectory_points, return_plane=True)
    if initial_plane is None:
        # Fallback to simple plane segmentation if up-vector detection fails
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(all_points)
        _, ground_indices = pcd_full.segment_plane(0.2, 3, 1000)
        return ground_indices

    # Use a spatial grid for efficient point lookup
    x_min, y_min = all_points[:, 0].min(), all_points[:, 1].min()
    grid_indices = np.floor((all_points[:, :2] - [x_min, y_min]) / grid_size).astype(int)
    point_grid = defaultdict(list)
    for i, grid_idx in enumerate(grid_indices):
        point_grid[tuple(grid_idx)].append(i)

    all_ground_indices = set()
    total_dist = np.sum(np.linalg.norm(np.diff(trajectory_points, axis=0), axis=1))
    num_steps = int(np.ceil(total_dist / step_size))
    if num_steps < 2:
        trajectory_indices = [0, len(trajectory_points) - 1] if len(trajectory_points) > 1 else [0]
    else:
        distances = np.cumsum(np.linalg.norm(np.diff(trajectory_points, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        interp_distances = np.linspace(0, distances[-1], num_steps)
        interp_indices = np.interp(interp_distances, distances, np.arange(len(trajectory_points)))
        trajectory_indices = np.unique(interp_indices.astype(int))

    for i in trajectory_indices:
        current_pose = trajectory_points[i]
        local_indices = []
        center_grid_idx = np.floor((current_pose[:2] - [x_min, y_min]) / grid_size).astype(int)
        search_grid_radius = int(np.ceil(search_radius / grid_size))
        for dx in range(-search_grid_radius, search_grid_radius + 1):
            for dy in range(-search_grid_radius, search_grid_radius + 1):
                local_indices.extend(point_grid.get((center_grid_idx[0] + dx, center_grid_idx[1] + dy), []))

        if len(local_indices) < 50: continue
        
        local_points = all_points[local_indices]
        
        ransac_points = local_points
        if len(local_points) > max_ransac_points:
            sample_indices = np.random.choice(len(local_points), max_ransac_points, replace=False)
            ransac_points = local_points[sample_indices]
        
        # This internal find_plane function is a simplified version for local refinement
        pcd_local = o3d.geometry.PointCloud()
        pcd_local.points = o3d.utility.Vector3dVector(ransac_points)
        plane_model, _ = pcd_local.segment_plane(0.2, 3, 50)

        if plane_model is not None:
            normal, d = plane_model[:3], plane_model[3]
            if np.dot(normal, up_vector) < 0:
                normal, d = -normal, -d
            if np.dot(normal, up_vector) < np.cos(np.radians(45)): continue
            
            signed_dist_to_pose = np.dot(normal, current_pose) + d
            if not (1.0 < signed_dist_to_pose < 3.0): continue

            distances = np.abs(np.dot(local_points, normal) + d)
            inlier_mask_local = distances < 0.2
            original_inlier_indices = np.array(local_indices)[inlier_mask_local]
            all_ground_indices.update(original_inlier_indices)

    return np.array(list(all_ground_indices), dtype=int)

#!/usr/bin/env python3
import cv2
import numpy as np
import math
import open3d as o3d

def distance_point_to_plane_in_cb(board_pt, plane_model):
    a, b, c, d = plane_model
    numerator = a*board_pt[0] + b*board_pt[1] + c*board_pt[2] + d
    denominator = math.sqrt(a*a + b*b + c*c)
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator

def extract_pointcloud_from_depth(depth_image_m, intrinsics, depth_scale=1.0):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]


    depth_for_o3d = depth_image_m.astype(np.float32) / depth_scale
    o3d_depth = o3d.geometry.Image(depth_for_o3d)
    height, width = depth_for_o3d.shape
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        o3d_intrinsics,
        extrinsic=np.eye(4),
        depth_scale=1.0,    
        depth_trunc=5.0     
    )
    return pcd

def ransac_plane_fitting(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers

def plane_fit_checkerboard(depth_image_m,
                           corners2,
                           fx, fy, cx, cy,
                           distance_threshold=0.01,
                           ransac_n=3,
                           num_iterations=1000):
    pts_2d = corners2.reshape(-1, 2)
    min_x = int(np.min(pts_2d[:, 0]))
    max_x = int(np.max(pts_2d[:, 0]))
    min_y = int(np.min(pts_2d[:, 1]))
    max_y = int(np.max(pts_2d[:, 1]))
    
    mask = np.zeros_like(depth_image_m, dtype=np.uint8)
    cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), color=255, thickness=-1)
    masked_depth = depth_image_m.copy()
    masked_depth[mask == 0] = 0.0
    
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    pcd = extract_pointcloud_from_depth(masked_depth, intrinsics, depth_scale=1.0)
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if len(pcd.points) < 50:
        return None, None
    
    plane_model, inliers = ransac_plane_fitting(
        pcd,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers


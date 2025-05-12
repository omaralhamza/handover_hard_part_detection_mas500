#!/usr/bin/env python3
import cv2
import numpy as np
import math
import open3d as o3d

def distance_point_to_plane_in_cb(board_pt, plane_model):
    """
    Compute the (signed) distance from a 3D point (in checkerboard coordinates)
    to a plane given by (a, b, c, d), where the plane equation is:
       a*x + b*y + c*z + d = 0.
    """
    a, b, c, d = plane_model
    numerator = a*board_pt[0] + b*board_pt[1] + c*board_pt[2] + d
    denominator = math.sqrt(a*a + b*b + c*c)
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator

def extract_pointcloud_from_depth(depth_image_m, intrinsics, depth_scale=1.0):
    """
    Convert a numpy depth image (in meters) into an Open3D point cloud in camera coordinates.
    - depth_image_m: HxW float32 or float64 array with depth in meters.
    - intrinsics: dict with keys {fx, fy, cx, cy}, all in pixels.
    - depth_scale: scale factor (set to 1.0 if the depth image is already in meters).
    
    Returns: open3d.geometry.PointCloud (points in camera coordinates, in meters).
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Convert depth image to float32 (meters)
    depth_for_o3d = depth_image_m.astype(np.float32) / depth_scale

    o3d_depth = o3d.geometry.Image(depth_for_o3d)
    height, width = depth_for_o3d.shape
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        o3d_intrinsics,
        extrinsic=np.eye(4),
        depth_scale=1.0,    # Already in meters
        depth_trunc=5.0     # Truncate depths beyond 5 m
    )
    return pcd

def ransac_plane_fitting(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Perform RANSAC plane fitting on an Open3D point cloud.
    Returns a tuple (plane_model, inliers) where:
      - plane_model is [a, b, c, d] such that ax+by+cz+d=0 (all in meters),
      - inliers is a list of indices for points that support the plane.
    
    You may adjust the distance_threshold (in meters) and num_iterations to improve stability.
    """
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
    """
    Given the 2D checkerboard corner points and a depth image (in meters),
    this function:
      1) Computes the 2D bounding box of the checkerboard.
      2) Masks the depth image to keep only the checkerboard region.
      3) Converts the masked depth image to a point cloud.
      4) Optionally removes outlier points.
      5) Runs RANSAC to fit a plane to the checkerboard region.
    
    Parameters:
      - corners2: Nx1x2 or Nx2 array of detected checkerboard corner pixel coordinates.
      - depth_image_m: HxW depth image in meters.
      - fx, fy, cx, cy: camera intrinsics.
      - distance_threshold, ransac_n, num_iterations: RANSAC parameters (in meters).
    
    Returns:
      - plane_model: [a, b, c, d] of the fitted plane, or None if fitting fails.
      - inliers: indices of points that are inliers to the plane.
    """
    pts_2d = corners2.reshape(-1, 2)
    min_x = int(np.min(pts_2d[:, 0]))
    max_x = int(np.max(pts_2d[:, 0]))
    min_y = int(np.min(pts_2d[:, 1]))
    max_y = int(np.max(pts_2d[:, 1]))
    
    # Create a mask for the bounding box covering the checkerboard
    mask = np.zeros_like(depth_image_m, dtype=np.uint8)
    cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), color=255, thickness=-1)
    
    # Set depth=0 outside the bounding box
    masked_depth = depth_image_m.copy()
    masked_depth[mask == 0] = 0.0
    
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    pcd = extract_pointcloud_from_depth(masked_depth, intrinsics, depth_scale=1.0)
    
    # Remove statistical outliers to reduce noise around the checkerboard
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if len(pcd.points) < 50:
        # Not enough points to reliably fit a plane
        return None, None
    
    plane_model, inliers = ransac_plane_fitting(
        pcd,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers

# Example usage / test:
if __name__ == "__main__":
    # Example usage (won't run as-is unless you have an actual depth image
    # and corner detections). This is just for illustration.
    #
    # depth_image_m = cv2.imread("sample_depth.png", cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
    # corners2 = np.load("sample_corners.npy")  # Should be your checkerboard corners from cv2.findChessboardCorners
    #
    # fx, fy, cx, cy = 1349.4283, 1350.4358, 998.7163, 563.6248
    # plane_model, inliers = plane_fit_checkerboard(depth_image_m, corners2, fx, fy, cx, cy)
    # if plane_model is not None:
    #     print("Plane model:", plane_model)
    #     print("Number of inliers:", len(inliers))
    # else:
    #     print("Plane fitting failed.")
    pass

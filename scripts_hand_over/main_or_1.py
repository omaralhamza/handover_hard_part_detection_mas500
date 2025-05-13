#!/usr/bin/env python3
import sys
import time
import threading
import math
import os
import numpy as np
import cv2
import tkinter as tk
from collections import defaultdict

# ============================
# 1) Extra imports & logging code from your logging script
# ============================
import csv
import open3d as o3d
import pyrealsense2 as rs
from scipy.optimize import linear_sum_assignment
import apriltag  # NEW: Import the apriltag library

# ============================
# 2) Helpers from logging script
# ============================
def distance_point_to_plane_in_cb(point_cb, plane_cb):
    a, b, c, d = plane_cb
    num = a * point_cb[0] + b * point_cb[1] + c * point_cb[2] + d
    denom = math.sqrt(a*a + b*b + c*c)
    return num / denom if denom != 0 else 0.0

def get_average_depth(depth_frame, px, py, delta=3):
    vals = []
    for x in range(px - delta, px + delta + 1):
        for y in range(py - delta, py + delta + 1):
            try:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    vals.append(d)
            except RuntimeError:
                pass
    return sum(vals) / len(vals) if vals else 0.0

def pixel_to_camera(u, v, depth, cx_val, cy_val, fx_val, fy_val):
    if depth <= 0:
        return np.array([0,0,0], dtype=np.float32)
    X = (u - cx_val) * depth / fx_val
    Y = (v - cy_val) * depth / fy_val
    Z = depth 
    return np.array([X, Y, Z], dtype=np.float32)

def format_xyz(pt):
    return f"({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})"

def transform_to_checkerboard(cam_pt_m, rvec, tvec):
    # tvec in mm => convert to meters
    tvec_m = tvec.flatten() / 1000.0
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    world_pt = R_inv.dot((cam_pt_m - tvec_m).reshape(3,1))
    return world_pt.flatten()

def draw_checkerboard_axes(image, rvec, tvec, axis_length, mtx, dist_coeffs):
    axis_3d = np.float32([
        [0,0,0],
        [axis_length,0,0],
        [0,axis_length,0],
        [0,0,-axis_length]
    ])
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, mtx, dist_coeffs)
    origin = tuple(imgpts[0].ravel().astype(int))
    xend   = tuple(imgpts[1].ravel().astype(int))
    yend   = tuple(imgpts[2].ravel().astype(int))
    zend   = tuple(imgpts[3].ravel().astype(int))
    cv2.line(image, origin, xend, (0,0,255), 3)
    cv2.line(image, origin, yend, (0,255,0), 3)
    cv2.line(image, origin, zend, (255,0,0), 3)
    cv2.putText(image, "X", xend, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(image, "Y", yend, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(image, "Z", zend, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

def draw_crosshair_and_dashed_lines(image, dash_len=20, gap_len=10):
    h, w = image.shape[:2]
    ctr = (w//2, h//2)
    cv2.drawMarker(image, ctr, (255,255,255), cv2.MARKER_CROSS, 40, 2)
    x = 0
    while x < w:
        x_end = min(x+dash_len, w)
        cv2.line(image, (x, ctr[1]), (x_end, ctr[1]), (255,105,180), 2)
        x += dash_len+gap_len
    y = 0
    while y < h:
        y_end = min(y+dash_len, h)
        cv2.line(image, (ctr[0], y), (ctr[0], y_end), (255,105,180), 2)
        y += dash_len+gap_len

# Removed: assign_detection_names_13 and all related anchor naming functionality

def find_april_tag_corners(image, mtx, dist):
    """
    Detect AprilTags using the apriltag library.
    This function looks for an AprilTag with tag_id 22.
    If found, it returns (True, corners) where corners is a numpy array of shape (4,1,2).
    In practice, you should further adjust parameters and handle multiple detections as needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -- Only changed this part to remove quad_sigma, everything else is the same --
    options = apriltag.DetectorOptions(
        families='tag36h11',
        quad_decimate=2.0  # Keep downsample for fewer edges, removing 'quad_sigma'
    )
    detector = apriltag.Detector(options)

    try:
        detections = detector.detect(gray)
    except Exception as e:
        print(f"[AprilTag Error] Detection failed: {e}")
        return False, None

    for detection in detections:
        if detection.tag_id == 22:
            # detection.corners is (4,2); reshape to (4,1,2)
            corners = detection.corners.reshape((4,1,2)).astype(np.float32)
            return True, corners
    return False, None

# --------------------------------------------------------------------
# 3) The main runtime code (preserved exactly from your second script)
# --------------------------------------------------------------------

selection_active = False

# 1) Import our custom modules
from robot_ui_module import (
    initialize_robot,
    display_ui
)
from checkerboard_module import (
    find_checkerboard_pose,
    CHECKERBOARD,
    objp,
    square_size,
)
from yolo_module import run_yolo_detections
from solvepnp_helpers import rotationMatrixToEulerAngles
from draw_helpers import draw_crosshairs
from camera_realsense import initialize_camera, get_frames, stop_camera
from plane_fitting_module import plane_fit_checkerboard

# --------------------------------------------------------------------
# Query RealSense for factory intrinsics (1920×1080) + distortion
# --------------------------------------------------------------------
pipeline_intr = rs.pipeline()
cfg_intr      = rs.config()
cfg_intr.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
profile_intr  = pipeline_intr.start(cfg_intr)
intr          = profile_intr.get_stream(rs.stream.color) \
                          .as_video_stream_profile() \
                          .get_intrinsics()
pipeline_intr.stop()

fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy
mtx     = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
dist    = np.array(intr.coeffs, dtype=np.float32)

print(f"[INFO] Factory intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
print(f"[INFO] Distortion coeffs: {dist.tolist()}")

mtx = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float32)
dist = np.zeros(5, dtype=np.float32)

# Robot offsets (in mm)
ROBOT_OFFSET_X_MM = 11.0 * 10     # 113 mm
ROBOT_OFFSET_Y_MM = 14.5 * 10     # 147 mm

# --------------------------------------------------------------------
# Transformation from Board -> Robot
# --------------------------------------------------------------------
def world_to_robot_coords_mm(Xw_mm, Yw_mm, Zw_mm):
    """
    Converts 3D coordinates from the checkerboard frame (in mm) to the robot frame.
    It applies fixed offsets and does not force the Z value to be positive.
    """
    Xr_mm = Yw_mm + ROBOT_OFFSET_Y_MM
    Yr_mm = Xw_mm - ROBOT_OFFSET_X_MM
    offset = 0.0
    Zr_mm = Zw_mm - offset
    return Xr_mm, Yr_mm, Zr_mm

def get_tilt_angle(R):
    """
    Computes the tilt angle (in radians) from a rotation matrix.
    This is done by finding the angle between the camera’s z-axis (transformed)
    and the world’s z-axis.
    """
    cam_z = R.T @ np.array([0, 0, 1])
    theta = math.acos(np.clip(np.dot(cam_z, np.array([0, 0, 1])), -1.0, 1.0))
    return theta

# Helper for saving images
def save_image_in_folder(folder, image, prefix):
    """
    Saves the image in the specified folder under "live_processed_pictures".
    If more than 5 images exist, the oldest image is deleted.
    """
    base_dir = os.path.join("live_processed_pictures", folder)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg'))]
    if len(files) >= 5:
        files_full = [os.path.join(base_dir, f) for f in files]
        oldest_file = min(files_full, key=os.path.getctime)
        os.remove(oldest_file)
    filename = f"{prefix}_{int(time.time())}.png"
    full_path = os.path.join(base_dir, filename)
    cv2.imwrite(full_path, image)
    return full_path

# Optional: Debug function for Z computations
def debug_z_computations(label, raw_depth_m, avg_depth_m, Xc_m, Yc_m, Zc_m, tilt_mm, plane_mm, final_mm):
    """
    Prints detailed debug information about the Z computation for a detection.
    """
    print(f"[DEBUG] Label={label}")
    print(f"    Raw depth (center)   : {raw_depth_m:.4f} m")
    print(f"    avg_depth           : {avg_depth_m:.4f} m")
    print(f"    camera coords (Zc)  : {Zc_m:.4f} m")
    print(f"    old tilt-based Zw_mm: {tilt_mm:.3f} mm")
    print(f"    plane-based Zw_mm   : {plane_mm:.3f} mm")
    print(f"    FINAL Zw_mm         : {final_mm:.3f} mm\n")

# Optional: Debug function for printing snapshot info
def print_snapshot_debug(detections):
    """
    Prints a summary of each detection’s computed world coordinates.
    """
    print("\nSnapshot Debug Info for each detection:")
    for det in detections:
        lbl = det["hashtag"]
        (Xw, Yw, Zw) = det["world"]
        _, _, Zr = world_to_robot_coords_mm(Xw, Yw, Zw)
        print(f"  {lbl}: |Zw_mm| = {abs(Zw):.2f} mm, Robot Z = {Zr/1000:.3f} m")

# Robot Initialization and UI Functions
def initialize_robot():
    """
    Initializes the robot using Interbotix modules.
    """
    from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
    from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
    bot = InterbotixManipulatorXS(robot_model='vx300', group_name='arm', gripper_name='gripper')
    robot_startup()
    detecting_home_position(bot)
    return bot

def detecting_home_position(bot):
    """
    Sets the robot to its home (detecting) position and opens the gripper.
    """
    bot.arm.set_ee_pose_components(x=0.2, z=0.57)
    time.sleep(1)
    bot.gripper.release()

def move_robot_to_center(bot, center_x_m, center_y_m, center_z_m):
    """
    Moves the robot's end-effector to a target center position then returns it to the detecting position.
    """
    target_z = center_z_m
    print(f"[DEBUG] Moving to center => X={center_x_m:.3f}m, Y={center_y_m:.3f}m, Z={target_z:.3f}m")
    bot.arm.set_ee_pose_components(x=center_x_m, y=center_y_m, z=target_z+0.1)
    print("Returning to detecting position...")
    time.sleep(3)
    detecting_home_position(bot)

# Compute camera position in checkerboard (world) frame
def compute_camera_position_world(R, tvec):
    """
    Given the rotation matrix (R) and translation vector (tvec) from solvePnP,
    returns the camera position in the world (checkerboard) frame in mm,
    with the Z forced to be positive.
    """
    if tvec.shape == (3,):
        tvec = tvec.reshape((3,1))
    camera_pos = -R.T @ tvec  # in mm

    if camera_pos[2] < 0:
        camera_pos[2] = -camera_pos[2]

    return camera_pos.ravel()

# Main Function
def main():
    # --------------------------------------------------------------------
    # 1) Initialize Robot & Camera
    # --------------------------------------------------------------------
    bot = initialize_robot()
    pipeline, align = initialize_camera()

    # --------------------------------------------------------------------
    # 2) CSV Logging Setup (from logging code)
    # --------------------------------------------------------------------
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    csv_fn = script_name + ".csv"
    file_exists = os.path.isfile(csv_fn)
    csv_f = open(csv_fn, "a", newline="")
    csv_w = csv.writer(csv_f)

    # Write header only once
    if not file_exists:
        csv_w.writerow([
            "Timestamp",             # UNIX time (s)
            "Label",                 # Detection hashtag
            "PxCenter",              # Box center in pixels
            "Depth_m",               # Avg. depth at center (m)
            "Cam_fac_XYZ_m",         # Factory-camera 3D point (X,Y,Z) in CB frame (m)
            "Cam_hand_XYZ_m",        # Hand-camera 3D point    (X,Y,Z) in CB frame (m)
            "CB_fac_raw_XYZ_m",      # Checkerboard-frame raw fac (X,Y,Z) in m
            "CB_hand_raw_XYZ_m",     # Checkerboard-frame raw hand (X,Y,Z) in m
            "Rob_fac_ransac_XYZ_m",  # Robot-frame fac + RANSAC Z   (m)
            "Rob_hand_ransac_XYZ_m"  # Robot-frame hand + RANSAC Z  (m)
        ])


    save_folder = "/home/omar/Cameras/scripts/exp_2_ran"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Variables to hold the last computed pose
    last_rvec = None
    last_tvec = None
    last_R = None

    # Global plane model from RANSAC
    plane_model_global = None
    last_plane_time = time.time()
    plane_interval = 1.0

    # YOLO detection timing variables
    last_yolo_time = time.time()
    detection_interval = 0.5
    last_detections = []

    # New dictionaries for EMA smoothing
    ema_z = {}
    ema_plane_z = {}

    print("Press 's' to capture image (and show UI), 'u' to manually show UI,")
    print("Press 'w' to do the logging code and save the picture, 'q' to exit.")

    try:
        while True:
            # -------------------------------------------------------------
            # Fetch frames
            # -------------------------------------------------------------
            color_image, depth_image, depth_frame = get_frames(pipeline, align)
            if color_image is None or depth_image is None:
                continue

            # Save a raw copy for YOLO detection (unannotated)
            raw_frame = color_image.copy()

            # Convert the entire depth image to meters
            h, w = depth_image.shape
            depth_image_m = np.zeros_like(depth_image, dtype=np.float32)
            for v in range(h):
                for u in range(w):
                    depth_image_m[v, u] = depth_frame.get_distance(u, v)

            # -------------------------------------------------------------
            # Checkerboard and AprilTag Detection
            # -------------------------------------------------------------
            # Detect checkerboard pose and corners
            success_cb, rvec, tvec, corners2 = find_checkerboard_pose(color_image, mtx, dist)
            # Detect AprilTag corners using our new implementation
            success_ap, tag_corners = find_april_tag_corners(color_image, mtx, dist)

            # Update the pose if the checkerboard is detected
            if success_cb:
                last_rvec = rvec
                last_tvec = tvec
                last_R, _ = cv2.Rodrigues(rvec)
                # --- Draw the checkerboard axis (with labels) on the image ---
                draw_checkerboard_axes(color_image, rvec, tvec, 3 * square_size, mtx, dist)

            # -------------------------------------------------------------
            # Combine available corner detections for plane fitting
            # (Original approach: combine both checkerboard and AprilTag corners)
            if success_cb and success_ap:
                all_corners = np.vstack((corners2, tag_corners))
                plane_source = "c+a"
            elif success_cb:
                all_corners = corners2
                plane_source = "c"
            elif success_ap:
                all_corners = tag_corners
                plane_source = "a"
            else:
                all_corners = None
                plane_source = "None"

            # --- Draw golden markers for all detected corners ---
            if all_corners is not None:
                for corner in all_corners:
                    pt = tuple(corner.ravel().astype(int))
                    cv2.circle(color_image, pt, 5, (0,215,255), -1)

            # --- Display plane fitting source info in the top right corner (in larger font) ---
            text = "Plane Fitting: " + plane_source
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x = color_image.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(color_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 4)
            cv2.putText(color_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

            # -------------------------------------------------------------
            # Fit the plane using all available corners
            # -------------------------------------------------------------
            if all_corners is not None:
                now_plane = time.time()
                if (now_plane - last_plane_time) >= plane_interval:
                    try:
                        plane_model, inliers = plane_fit_checkerboard(
                            depth_image_m, all_corners, fx, fy, cx, cy,
                            distance_threshold=0.007,
                            ransac_n=6,
                            num_iterations=1000
                        )
                        if plane_model is not None:
                            a_c, b_c, c_c, d_c = plane_model
                            norm_abc = math.sqrt(a_c**2 + b_c**2 + c_c**2)
                            plane_model_global = (a_c, b_c, c_c, d_c, norm_abc)
                        else:
                            plane_model_global = None
                    except Exception:
                        plane_model_global = None
                    last_plane_time = now_plane

            # -------------------------------------------------------------
            # YOLO detection & 3D coordinate computation
            # -------------------------------------------------------------
            # Run YOLO detection on the raw image (to avoid interference from annotations)
            now = time.time()
            if now - last_yolo_time >= detection_interval:
                raw_dets = run_yolo_detections(raw_frame, depth_frame, mtx, dist, last_R, last_tvec)
                new_dets = []
                for det in raw_dets:
                    x1, y1, x2, y2 = det["box"]
                    cx_box, cy_box = det["center"]
                    hashtag = det["hashtag"]

                    # average depth
                    delta = 2
                    y1_roi = max(0, cy_box - delta)
                    y2_roi = min(h, cy_box + delta + 1)
                    x1_roi = max(0, cx_box - delta)
                    x2_roi = min(w, cx_box + delta + 1)
                    roi = depth_image_m[y1_roi:y2_roi, x1_roi:x2_roi]
                    valid_depths = roi[roi > 0]
                    avg_depth = valid_depths.mean() if valid_depths.size > 0 else depth_frame.get_distance(cx_box, cy_box)
                    det["avg_depth_m"] = avg_depth

                    # raw camera coords
                    Xc_m = (cx_box - cx) * (avg_depth / fx)
                    Yc_m = (cy_box - cy) * (avg_depth / fy)
                    Zc_m = avg_depth

                    # z_perp (same as avg_depth)
                    z_perp = avg_depth
                    Xc_m_corr = (cx_box - cx) * (z_perp / fx)
                    Yc_m_corr = (cy_box - cy) * (z_perp / fy)
                    Zc_m_corr = z_perp

                    # board coords
                    if (last_R is not None) and (last_tvec is not None):
                        R_inv = np.linalg.inv(last_R)
                        tvec_m = last_tvec / 1000.0
                        p_cam_corr = np.array([[Xc_m_corr], [Yc_m_corr], [Zc_m_corr]], dtype=np.float32)
                        p_world_corr = R_inv @ (p_cam_corr - tvec_m)
                        Xw_mm_corr = p_world_corr[0][0] * 1000.0
                        Yw_mm_corr = p_world_corr[1][0] * 1000.0
                        Zw_mm_corr = p_world_corr[2][0] * 1000.0
                        Zw_mm_corr = -Zw_mm_corr  # Flip sign for z_perp
                    else:
                        Xw_mm_corr = Xc_m_corr * 1000.0
                        Yw_mm_corr = Yc_m_corr * 1000.0
                        Zw_mm_corr = -Zc_m_corr * 1000.0

                    # EMA smoothing for raw depth value
                    alpha = 0.75
                    if hashtag not in ema_z:
                        ema_z[hashtag] = Zw_mm_corr
                    else:
                        ema_z[hashtag] = alpha * Zw_mm_corr + (1 - alpha) * ema_z[hashtag]
                    avg_Zw_mm = ema_z[hashtag]

                    # tilt-based fallback
                    theta = get_tilt_angle(last_R) if last_R is not None else 0.0
                    k = -1.5
                    tilt_adjust_m = k * (1 - math.cos(theta))
                    tilt_adjust_mm = tilt_adjust_m * 1000.0
                    zw_tilt_mm = avg_Zw_mm + tilt_adjust_mm

                    # plane-fitting if available, with EMA smoothing on plane-based depth
                    final_Zw_mm = zw_tilt_mm
                    if plane_model_global is not None:
                        a_c, b_c, c_c, d_c, norm_abc = plane_model_global
                        numerator = (a_c*Xc_m + b_c*Yc_m + c_c*Zc_m + d_c)
                        plane_dist_m = numerator / (norm_abc + 1e-12)
                        zw_plane_mm = plane_dist_m * 1000.0

                        if hashtag not in ema_plane_z:
                            ema_plane_z[hashtag] = zw_plane_mm
                        else:
                            ema_plane_z[hashtag] = alpha * zw_plane_mm + (1 - alpha) * ema_plane_z[hashtag]
                        smoothed_plane_mm = ema_plane_z[hashtag]
                        final_Zw_mm = -smoothed_plane_mm

                    # Do not force final_Zw_mm to be positive so that objects under the plane are preserved.
                    # final_Zw_mm = abs(final_Zw_mm)

                    # store coords
                    det["world_zperp"] = (Xw_mm_corr, Yw_mm_corr, Zw_mm_corr)
                    det["robot_zperp"] = world_to_robot_coords_mm(Xw_mm_corr, Yw_mm_corr, Zw_mm_corr)

                    det["world"] = (Xw_mm_corr, Yw_mm_corr, final_Zw_mm)
                    det["robot"] = world_to_robot_coords_mm(Xw_mm_corr, Yw_mm_corr, final_Zw_mm)
                    


                    # --------------------------------------------------------------------
                    # NEW CODE: Compute robot coordinates for the bounding box corners
                    # --------------------------------------------------------------------
                    box_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    robot_corners = []
                    for (u, v) in box_points:
                        depth_corner = get_average_depth(depth_frame, u, v, delta=2)
                        cam_pt_corner = pixel_to_camera(u, v, depth_corner, cx, cy, fx, fy)
                        if (last_R is not None) and (last_tvec is not None):
                            R_inv = np.linalg.inv(last_R)
                            tvec_m = last_tvec / 1000.0
                            p_cam_corner = np.array([[cam_pt_corner[0]], [cam_pt_corner[1]], [cam_pt_corner[2]]], dtype=np.float32)
                            p_world_corner = R_inv @ (p_cam_corner - tvec_m)
                            Xw_mm_corner = p_world_corner[0][0] * 1000.0
                            Yw_mm_corner = p_world_corner[1][0] * 1000.0
                            Zw_mm_corner = p_world_corner[2][0] * 1000.0
                        else:
                            Xw_mm_corner = cam_pt_corner[0] * 1000.0
                            Yw_mm_corner = cam_pt_corner[1] * 1000.0
                            Zw_mm_corner = cam_pt_corner[2] * 1000.0
                        robot_corner_mm = world_to_robot_coords_mm(
                            Xw_mm_corner, Yw_mm_corner, Zw_mm_corner
                        )
                        # immediately convert to metres for the robot
                        robot_corner_m = tuple(v/1000.0 for v in robot_corner_mm)
                        robot_corners.append(robot_corner_m)
                    det["corners_robot"] = robot_corners
                    # --------------------------------------------------------------------

                    new_dets.append(det)
                last_detections = new_dets
                last_yolo_time = now

            # -------------------------------------------------------------
            # Draw bounding boxes and info
            # -------------------------------------------------------------
            for det in last_detections:
                x1, y1, x2, y2 = det["box"]
                hashtag = det["hashtag"]
                conf = det["conf"]
                cx_box, cy_box = det["center"]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_txt = f"{hashtag}: {conf:.2f}"
                cv2.putText(color_image, label_txt, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.drawMarker(color_image, (cx_box, cy_box), (0, 255, 0),
                               cv2.MARKER_CROSS, 10, 2)

                (Xr_zp, Yr_zp, Zr_zp) = det["robot_zperp"]
                zperp_str = f"Z_perp: X={Xr_zp/1000:.3f}, Y={Yr_zp/1000:.3f}, Z={Zr_zp/1000:.3f} m"
                cv2.putText(color_image, zperp_str, (x1, y1+60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                (Xw_p, Yw_p, Zw_p) = det["world"]
                (Xr_p, Yr_p, Zr_p) = det["robot"]
                plane_str = f"Plane: X={Xr_p/1000:.3f}, Y={Yr_p/1000:.3f}, Z={Zr_p/1000:.3f} m"
                cv2.putText(color_image, plane_str, (x1, y1+85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # -------------------------------------------------------------
            # Display the stream
            # -------------------------------------------------------------
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            scale_factor = 0.3
            small_color = cv2.resize(color_image, (0,0), fx=scale_factor, fy=scale_factor)
            small_depth = cv2.resize(depth_colormap, (0,0), fx=scale_factor, fy=scale_factor)
            combined_view = np.hstack((small_color, small_depth))
            cv2.imshow("RGB + Depth Stream", combined_view)

            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key).lower()

            # -------------------------------------------------------------
            # 's' = Snapshot code (original)
            # -------------------------------------------------------------
            if key_char == 's':
                raw_img  = raw_frame.copy()
                yolo_img = color_image.copy()

                # save raw (no annotations)
                raw_path  = save_image_in_folder("raw", raw_img,  "raw")
                # save with YOLO & checkerboard annotations
                yolo_path = save_image_in_folder("yolo", yolo_img, "yolo")

                print(f"[INFO] Raw image saved:  {raw_path}")
                print(f"[INFO] YOLO image saved: {yolo_path}\n")

                if plane_model_global is not None:
                    a_c, b_c, c_c, d_c, norm_abc = plane_model_global
                    camera_distance = abs(d_c) / (norm_abc + 1e-12)
                    normal_cam = np.array([a_c, b_c, c_c]) / (norm_abc + 1e-12)
                    normal_world = last_R.T @ normal_cam if last_R is not None else None
                    print(f"[DEBUG] Plane: {a_c:.3f}x + {b_c:.3f}y + {c_c:.3f}z + {d_c:.3f} = 0, "
                          f"norm={norm_abc:.6f}, distance={camera_distance:.3f} m, "
                          f"normal_cam={normal_cam}, normal_world={normal_world}")
                    print(f"[INFO] Valid plane: distance={camera_distance:.2f} m")

                if last_R is not None:
                    theta = get_tilt_angle(last_R)
                    k = -1.5
                    tilt_adjust_m = k * (1 - math.cos(theta))
                    tilt_adjust_mm = tilt_adjust_m * 1000.0
                    print(f"[TILT] Theta: {theta:.3f} rad, Tilt adjustment: {tilt_adjust_m:.4f} m ({tilt_adjust_mm:.1f} mm)")
                else:
                    print("[TILT] No valid rotation available to calculate tilt.")

                gray_s = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                ret_s, corners_s = cv2.findChessboardCorners(gray_s, CHECKERBOARD, None)
                if ret_s:
                    print("[INFO] Checkerboard detected in raw snapshot.")
                    corners2_s = cv2.cornerSubPix(
                        gray_s, corners_s, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    success_s, rvec_s, tvec_s = cv2.solvePnP(objp, corners2_s, mtx, dist)
                    if success_s:
                        R_s, _ = cv2.Rodrigues(rvec_s)
                        print("\n=== Extrinsic Matrix ===")
                        print(np.hstack([R_s, tvec_s]))
                        print("\n=== Rotation Matrix ===")
                        print(R_s)
                        print("\n=== Translation Vector (mm) ===")
                        print(f"X: {tvec_s[0][0]:.2f}, Y: {tvec_s[1][0]:.2f}, Z: {tvec_s[2][0]:.2f}")
                        roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R_s)
                        print(f"Roll: {roll_deg:.2f}, Pitch: {pitch_deg:.2f}, Yaw: {yaw_deg:.2f}")

                        cam_pos_snap = compute_camera_position_world(R_s, tvec_s)
                        print("\n===Camera position in Checkerboard Frame ===")
                        print(f"    X: {cam_pos_snap[0]:.2f} mm, Y: {cam_pos_snap[1]:.2f} mm, Z: {cam_pos_snap[2]:.2f} mm")
                    else:
                        print("[ERROR] solvePnP failed on raw snapshot.")
                else:
                    print("[ERROR] Checkerboard not detected in raw snapshot.")

                print("\nBBox centers at snapshot time:")
                if last_detections:
                    for det in last_detections:
                        lbl = det["hashtag"]
                        (Xw, Yw, Zw) = det["world"]
                        (Xw_zp, Yw_zp, Zw_zp) = det["world_zperp"]
                        avg_depth = det.get("avg_depth_m", 0.0)

                        if Xw is not None:
                            Xr_mm, Yr_mm, Zr_mm = world_to_robot_coords_mm(Xw, Yw, Zw)
                            print(f"  {lbl} => Robot=({Xr_mm/1000:.3f}, {Yr_mm/1000:.3f}, {Zr_mm/1000:.3f} m)")
                        else:
                            print(f"  {lbl} => 3D not available")

                        print(f"\n[INFO] For {lbl}:")
                        print(f"    Raw depth (Z from avg_depth): {avg_depth:.4f} m")
                        z_perp_m = Zw_zp / 1000.0
                        print(f"    Corrected (z_perp) depth:     {z_perp_m:.4f} m")
                        print(f"    RANSAC/tilt-based height:     {Zw/1000:.4f} m")
                        print(f"[DEBUG] World coords (z_perp based): X={Xw_zp/1000:.4f} m, Y={Yw_zp/1000:.4f} m, Z={Zw_zp/1000:.4f} m")
                        print(f"[DEBUG] World coords (RANSAC/tilt based): X={Xw/1000:.4f} m, Y={Yw/1000:.4f} m, Z={Zw/1000:.4f} m")

                else:
                    print("[INFO] No detections to show in snapshot.")

                # UI with detection details
                if last_detections:
                    detection_tuples = []
                    for det in last_detections:
                        lbl = det["hashtag"]
                        cname = det["class_name"]
                        cx_box, cy_box = det["center"]
                        conf = det["conf"]
                        (Xw, Yw, Zw) = det["world"]
                        if Xw is not None:
                            Xr_mm, Yr_mm, Zr_mm = world_to_robot_coords_mm(Xw, Yw, Zw)
                            robot_x_m = Xr_mm/1000.
                            robot_y_m = Yr_mm/1000.
                            robot_z_m = Zr_mm/1000.
                        else:
                            robot_x_m = robot_y_m = robot_z_m = 0.
                        corners_m = det.get("corners_robot", [])
                        robot_info = f"X:{robot_x_m:.3f}, Y:{robot_y_m:.3f}, Z:{robot_z_m:.3f} m"
                        detection_tuples.append((lbl, cname, cx_box, cy_box, conf,
                                                 robot_x_m, robot_y_m, robot_z_m,
                                                 corners_m, robot_info))
                    threading.Thread(target=display_ui, args=(detection_tuples, bot), daemon=True).start()
                else:
                    print("[INFO] No detection results available for UI.")



            # -------------------------------------------------------------
            # 'w' = Logging code integration
            # -------------------------------------------------------------
            # -------------------------------------------------------------
            elif key_char == 'w':
                # 1) Check that we have a valid checkerboard pose and plane
                if last_rvec is None or last_tvec is None or plane_model_global is None:
                    print("[WARN] Checkerboard or plane not valid -> skip log.")
                    continue
                # 2) Save combined RGB+depth image
                now_ts = time.time()
                fname  = f"{script_name}_{int(now_ts)}.png"
                outp   = os.path.join(save_folder, fname)
                clipped = np.clip(depth_image.astype(np.float32), 0.3, 1.5)
                normed  = ((clipped - 0.3)/(1.5-0.3)*255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
                cv2.imwrite(outp, np.hstack((color_image, depth_vis)))
                print(f"[INFO] Saved image {outp} for logging.")

                # 3) Hand-camera intrinsics
                hand_fx, hand_fy = 1340.15318, 1342.81937
                hand_cx, hand_cy = 975.419561, 587.808154

                # 4) Per-detection logging
                for det in last_detections:
                    # a) Pixel center + depth
                    x1, y1, x2, y2 = det["box"]
                    ctr_px  = ((x1+x2)//2, (y1+y2)//2)
                    depth_m = get_average_depth(depth_frame, ctr_px[0], ctr_px[1], delta=1)

                    # b) Back-project to both cameras (metres)
                    cam_fac  = pixel_to_camera(ctr_px[0], ctr_px[1],
                                            depth_m, cx, cy, fx, fy)
                    cam_hand = pixel_to_camera(ctr_px[0], ctr_px[1],
                                            depth_m,
                                            hand_cx, hand_cy,
                                            hand_fx, hand_fy)

                    # c) Transform into checkerboard frame
                    cb_fac  = transform_to_checkerboard(cam_fac,  last_rvec, last_tvec)
                    cb_hand = transform_to_checkerboard(cam_hand, last_rvec, last_tvec)
                    cb_fac[2], cb_hand[2] = abs(cb_fac[2]), abs(cb_hand[2])

                    # d) RANSAC Z in mm; factory XY from det["world"], hand XY from cb_hand
                    Xw_mm_fac, Yw_mm_fac, Zr_mm = det["world"]
                    Xw_mm_hand = cb_hand[0] * 1000.0
                    Yw_mm_hand = cb_hand[1] * 1000.0

                    # Compute robot-frame points (metres)
                    rob_fac  = tuple(v/1000.0 for v in
                                    world_to_robot_coords_mm(Xw_mm_fac, Yw_mm_fac, Zr_mm))
                    rob_hand = tuple(v/1000.0 for v in
                                    world_to_robot_coords_mm(Xw_mm_hand, Yw_mm_hand, Zr_mm))

                    # e) Write to CSV
                    row = [
                        now_ts,
                        det["hashtag"],
                        str(ctr_px),
                        f"{depth_m:.3f}",
                        format_xyz(cam_fac),
                        format_xyz(cam_hand),
                        format_xyz(cb_fac),
                        format_xyz(cb_hand),
                        format_xyz(rob_fac),
                        format_xyz(rob_hand),
                    ]
                    csv_w.writerow(row)
                    csv_f.flush()

                    # f) Print the exact same row to terminal
                    print("Logged:", 
                        f"Time={row[0]:.5f},",
                        f"Tag={row[1]},",
                        f"Px={row[2]},",
                        f"Depth={row[3]} m,",
                        f"Cam_fac={row[4]},",
                        f"Cam_hand={row[5]},",
                        f"CB_fac_raw={row[6]},",
                        f"CB_hand_raw={row[7]},",
                        f"Rob_fac_ransac={row[8]},",
                        f"Rob_hand_ransac={row[9]}")



            # -------------------------------------------------------------
            # 'u' = Manual UI Display
            # -------------------------------------------------------------
            elif key_char == 'u':
                if last_detections:
                    detection_tuples = []
                    for det in last_detections:
                        lbl = det["hashtag"]
                        cname = det["class_name"]
                        cx_box, cy_box = det["center"]
                        conf = det["conf"]
                        (Xw, Yw, Zw) = det["world"]
                        if Xw is not None:
                            Xr_mm, Yr_mm, Zr_mm = world_to_robot_coords_mm(Xw, Yw, Zw)
                            robot_x_m = Xr_mm/1000.
                            robot_y_m = Yr_mm/1000.
                            robot_z_m = Zr_mm/1000.
                        else:
                            robot_x_m = robot_y_m = robot_z_m = 0.
                        corners_m = det.get("corners_robot", [])
                        robot_info = f"X:{robot_x_m:.3f}, Y:{robot_y_m:.3f}, Z:{robot_z_m:.3f} m"
                        detection_tuples.append((lbl, cname, cx_box, cy_box, conf,
                                                 robot_x_m, robot_y_m, robot_z_m,
                                                 corners_m, robot_info))
                    threading.Thread(target=display_ui, args=(detection_tuples, bot), daemon=True).start()
                else:
                    print("[INFO] No detection results available for UI.")

            # -------------------------------------------------------------
            # 'q' = Exit
            # -------------------------------------------------------------
            elif key_char == 'q':
                print("[INFO] Exiting...")
                break

            cv2.waitKey(1)

    finally:
        # Clean up
        stop_camera(pipeline)
        cv2.destroyAllWindows()
        csv_f.close()
        # Optionally, shut down the robot if needed.

if __name__=="__main__":
    main()
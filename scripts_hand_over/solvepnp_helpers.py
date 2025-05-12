import cv2
import numpy as np
from draw_helpers import draw_dashed_line

def rotationMatrixToEulerAngles(R):
    sy = (R[0, 0]**2 + R[1, 0]**2)**0.5
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

def recheck_solvePnP_on_raw(
    raw_color_copy,
    color_image,
    mtx,
    dist,
    objp,
    criteria,
    square_size,
    CAMERA_HEIGHT
):
    """
    Re-check solvePnP on 'raw_color_copy' (un-annotated).
    If success, draws the axes + dashed lines on 'color_image'.
    Returns True if success, False otherwise.
    """
    gray_for_s = cv2.cvtColor(raw_color_copy, cv2.COLOR_BGR2GRAY)
    ret_s, corners_s = cv2.findChessboardCorners(gray_for_s, (9, 6), None)
    if not ret_s:
        print("[ERROR] Checkerboard not detected.")
        return False

    corners2_s = cv2.cornerSubPix(gray_for_s, corners_s, (11, 11), (-1, -1), criteria)
    success_s, rvec_s, tvec_s = cv2.solvePnP(objp, corners2_s, mtx, dist)
    if not success_s:
        print("[ERROR] solvePnP failed.")
        return False

    # Draw axes
    axis_3d_s = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3) * square_size
    imgpts_s, _ = cv2.projectPoints(axis_3d_s, rvec_s, tvec_s, mtx, dist)
    origin_s = tuple(corners2_s[0].ravel().astype(int))
    x_pt_s = tuple(imgpts_s[0].ravel().astype(int))
    y_pt_s = tuple(imgpts_s[1].ravel().astype(int))
    z_pt_s = tuple(imgpts_s[2].ravel().astype(int))

    cv2.line(color_image, origin_s, x_pt_s, (0,0,255), 5)
    cv2.line(color_image, origin_s, y_pt_s, (0,255,0), 5)
    cv2.line(color_image, origin_s, z_pt_s, (255,0,0), 5)

    cv2.putText(color_image, "X", x_pt_s, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(color_image, "Y", y_pt_s, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(color_image, "Z", z_pt_s, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    # Extend X, Y with dashed lines
    extended_len = 8
    Xplus_3D  = np.float32([[ extended_len,  0,  0]]).reshape(-1,3)*square_size
    Xminus_3D = np.float32([[-extended_len,  0,  0]]).reshape(-1,3)*square_size
    Yplus_3D  = np.float32([[0,  extended_len,  0]]).reshape(-1,3)*square_size
    Yminus_3D = np.float32([[0, -extended_len,  0]]).reshape(-1,3)*square_size

    xplus_2d,  _ = cv2.projectPoints(Xplus_3D,  rvec_s, tvec_s, mtx, dist)
    xminus_2d, _ = cv2.projectPoints(Xminus_3D, rvec_s, tvec_s, mtx, dist)
    yplus_2d,  _ = cv2.projectPoints(Yplus_3D,  rvec_s, tvec_s, mtx, dist)
    yminus_2d, _ = cv2.projectPoints(Yminus_3D, rvec_s, tvec_s, mtx, dist)

    xplus_2d  = tuple(xplus_2d[0].ravel().astype(int))
    xminus_2d = tuple(xminus_2d[0].ravel().astype(int))
    yplus_2d  = tuple(yplus_2d[0].ravel().astype(int))
    yminus_2d = tuple(yminus_2d[0].ravel().astype(int))

    draw_dashed_line(color_image, x_pt_s, xplus_2d,  color=(0,0,255), thickness=3, num_dashes=10)
    draw_dashed_line(color_image, origin_s, xminus_2d, color=(0,0,255), thickness=3, num_dashes=10)
    draw_dashed_line(color_image, y_pt_s, yplus_2d,  color=(0,255,0), thickness=3, num_dashes=10)
    draw_dashed_line(color_image, origin_s, yminus_2d, color=(0,255,0), thickness=3, num_dashes=10)

    R_s, _ = cv2.Rodrigues(rvec_s)
    extrinsic_matrix = np.hstack([R_s, tvec_s])
    print("\n=== Extrinsic Matrix ===\n", extrinsic_matrix)
    print("\n=== Rotation Matrix ===\n", R_s)
    print("\n=== Translation Vector (mm) ===")
    print(f"X: {tvec_s[0][0]:.2f}, Y: {tvec_s[1][0]:.2f}, Z: {tvec_s[2][0]:.2f}")

    roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R_s)
    print(f"\n=== Rotation Angles (degrees) ===\nRoll: {roll_deg:.2f}, Pitch: {pitch_deg:.2f}, Yaw: {yaw_deg:.2f}")

    R_inv_s = np.linalg.inv(R_s)
    camera_position_s = -R_inv_s @ tvec_s
    print(f"\n=== Camera Position in World Frame ===")
    print(f"X: {camera_position_s[0][0]:.2f}, Y: {camera_position_s[1][0]:.2f}, Z: {abs(camera_position_s[2][0]):.2f} mm")
    print(f"Expected CAMERA_HEIGHT = {CAMERA_HEIGHT} mm")

    # Fix sign if negative
    if camera_position_s[2][0] < 0:
        print("\n⚠️ Z-Axis is inverted! Fixing it...")
        camera_position_s[2][0] = abs(camera_position_s[2][0])

    height_error = abs(camera_position_s[2][0] - CAMERA_HEIGHT)
    print(f"[Check] Camera height error = {height_error:.2f} mm")

    return True
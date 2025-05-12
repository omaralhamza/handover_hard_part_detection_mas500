import numpy as np
import cv2

# Checkerboard pattern and square size (in mm)
CHECKERBOARD = (9, 6)
square_size  = 26  # mm


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Precompute 3D object points for the checkerboard, in mm
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

def find_checkerboard_pose(color_image, mtx, dist):
    """
    Finds the checkerboard corners in 'color_image' and runs solvePnP.
    Returns:
      (True, rvec, tvec, corners2) if success,
      otherwise (False, None, None, None).
    tvec is in mm because our 'objp' uses mm.
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return False, None, None, None

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    if not success:
        return False, None, None, None

    return True, rvec, tvec, corners2

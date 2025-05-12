#%% load modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#%% termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6, 3), np.float32)  # 6*9,3
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane.

# Update the path to your images on Ubuntu
images = glob.glob('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/*.jpg')

# Initialize gray variable
gray = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        imgpoints.append(corners)

        # Naming a window
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        # Using resizeWindow()
        cv2.resizeWindow("Resized_Window", 1920, 1080)
        
        # Displaying the image
        img_draw = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow("Resized_Window", img_draw)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# Make sure we have points for calibration before calling calibrateCamera
if len(objpoints) > 0 and len(imgpoints) > 0 and gray is not None:
    objpoints = np.array(objpoints)
    imgpoints = np.array(imgpoints)

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save calibration parameters
    np.savez('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/C.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("Not enough points to perform calibration.")
    
    
print(mtx)
#%% extrinsic calibration

def draw(img, corners, imgpts):
    corner = (tuple(corners[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# Load previously saved data
with np.load('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/C.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane.

images = glob.glob('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/*.jpg')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Find the rotation and translation vectors.
        ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        corners2 = corners2.astype(int)
        imgpts = imgpts.astype(int)

        # Naming a window
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 1800, 1080)

        # Displaying the image
        img = draw(img, corners2, imgpts)
        cv2.imshow("Resized_Window", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#print tvec in x, y, z direction for each image
print("Translation Vectors (tvec):")
for tvec in tvecs:
    print(tvec)

# print rotation in x, y, z direction for each image
print("Rotation Vectors (rvec):")
for rvec in rvecs:
    print(rvec)

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6, 3), np.float32) 
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


objpoints = []  
imgpoints = []  


images = glob.glob('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/*.jpg')
gray = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        imgpoints.append(corners)

        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 1920, 1080)
        img_draw = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow("Resized_Window", img_draw)
        cv2.waitKey(200)

cv2.destroyAllWindows()

if len(objpoints) > 0 and len(imgpoints) > 0 and gray is not None:
    objpoints = np.array(objpoints)
    imgpoints = np.array(imgpoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savez('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/C.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("Not enough points to perform calibration.")
    
    
print(mtx)


def draw(img, corners, imgpts):
    corner = (tuple(corners[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

with np.load('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/C.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)

objpoints = []  
imgpoints = [] 

images = glob.glob('/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480/*.jpg')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        corners2 = corners2.astype(int)
        imgpts = imgpts.astype(int)
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 1800, 1080)
        img = draw(img, corners2, imgpts)
        cv2.imshow("Resized_Window", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
print("Translation Vectors (tvec):")
for tvec in tvecs:
    print(tvec)
print("Rotation Vectors (rvec):")
for rvec in rvecs:
    print(rvec)

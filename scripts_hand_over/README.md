## Module Descriptions

### `main_or_1.py`  
**Entry point** for the entire pipeline.  
- **`main()`**  
  - Instantiates the following components:  
    - `RealSenseCamera` (camera_realsense.py)  
    - `CheckerboardDetector` (checkerboard_module.py)  
    - `AprilTagDetector` (apriltag_module.py) if checkerboard fails  
    - `YOLODetector` (yolo_module.py)  
    - `RobotUI` (robot_ui_module.py)  
  - **Loop**:  
    1. `color, depth = camera.read_frames()`  
    2. `success_cb, cb_corners, rvec, tvec = checker.find_checkerboard_pose(color, mtx, dist)`  
    3. If `not success_cb`, fallback to AprilTag:  
       `success_ap, ap_corners = apriltag.find_april_tag_corners(color, mtx, dist)`  
    4. `detections = yolo.run_yolo_detections(color, depth, mtx, dist)`  
    5. Combine all corner data and 3D points, call `ui.display_ui(detections, bot)`  
    6. Send robot commands based on user selection  
  - Handles graceful shutdown: `camera.stop()`, `cv2.destroyAllWindows()`

---

### `camera_realsense.py`  
**RealSense camera interface** using `pyrealsense2`.  
- **`__init__(self, w=640, h=480, fps=30)`**  
  - Configures and starts the pipeline  
  - Enables aligned depth + color streams  
- **`read_frames(self) -> (color_image: np.ndarray, depth_frame)`**  
  - Waits for frames, aligns depth to color, returns BGR image + raw depth frame  
  - On failure returns `(None, None)`  
- **`stop(self)`**  
  - Stops streaming and cleans up the pipeline  

---

### `check_intrnsic.py`  
**Utility** to inspect factory intrinsics.  
- **`print_intrinsics()`**  
  - Fetches the RealSense intrinsics for the color stream  
  - Prints:  
    - Image size, principal point `(ppx, ppy)`  
    - Focal lengths `(fx, fy)`  
    - Distortion coefficients and model  

---

### `updated_realsense_cam_capture.py`  
**Calibration image capture tool**.  
- **`main()`**  
  - Opens a live color stream window  
  - **Hotkeys**:  
    - `s`: Save current frame (e.g. for calibration)  
    - `q`: Exit capture  
  - Saves images to a specified folder for later chessboard detection  

---

### `cam_cal_class.py`  
**Chessboard camera calibration** using OpenCV.  
- **`CameraCalibration(pattern=(9,6), square_size=0.025)`**  
  - `pattern`: # inner corners (cols, rows)  
  - `square_size`: side length in meters  
- **`find_image_points(image_files: List[str]) -> (objpoints, imgpoints)`**  
  - Detects chessboard in each image, refines corners  
- **`calibrate()`**  
  - Calls `cv2.calibrateCamera()`  
  - Saves `mtx`, `dist`, `rvecs`, `tvecs` to `.npz`  
- **`visualize_axes(image, rvec, tvec)`**  
  - Projects a 3D axis onto the calibration image for verification  

---

### `checkerboard_module.py`  
**Pose estimation** using a known checkerboard.  
- **`find_checkerboard_pose(color_image, mtx, dist) -> (success, corners, rvec, tvec)`**  
  - Detects 9×6 pattern with `cv2.findChessboardCorners()`  
  - Refines with `cv2.cornerSubPix()`  
  - Solves PnP: `cv2.solvePnP()`  
  - Returns `True` and corner arrays + pose vectors on success  

---

### `apriltag_module.py`  
**AprilTag fallback detector**.  
- **`AprilTagDetector()`**  
  - Wraps `apriltag.Detector()`  
- **`detect(self, gray_image, depth_frame) -> List[dict]`**  
  - For each tag:  
    - Reads `tag_id`, `corners`, `center`  
    - Queries depth at center pixel  
    - Returns list of `{ id, center, depth, corners }`  
- **`draw_on(self, image, detections) -> image`**  
  - Renders tag outlines, centers, and IDs on the color image  

---

### `plane_fitting_module.py`  
**3D plane fitting** for checkerboard points.  
- **`fit_plane_to_points(points: np.ndarray) -> (plane_model)`**  
  - Uses least-squares or Open3D RANSAC to fit plane  
- **`distance_point_to_plane(point: np.ndarray, plane_model) -> float`**  
  - Computes signed distance from a 3D point to the plane  

---

### `solvepnp_helpers.py`  
**PnP and visualization utilities**.  
- **`rotationMatrixToEulerAngles(R) -> (roll, pitch, yaw)`**  
- **`print_camera_pose(R, tvec)`**  
  - Converts to world coordinates and prints camera height/tilt  
- **`draw_axes(image, mtx, dist, rvec, tvec)`**  
  - Calls `cv2.drawFrameAxes()` to overlay coordinate axes  

---

### `yolo_module.py`  
**YOLOv8 object detection** wrapper.  
- **`YOLODetector(model_path: str)`**  
  - Loads Ultralytics `best.pt`  
- **`run_yolo_detections(color_image, depth_frame, mtx, dist, classes=None, conf_thresh=0.5) -> List[dict]`**  
  - Performs inference, filters by class and confidence  
  - Projects 2D detections into 3D using depth and intrinsics  
  - Returns list of `{ bbox, confidence, class_id, center_2d, center_3d, corners_3d }`  

---

### `draw_helpers.py`  
**Visualization helpers**.  
- **`draw_crosshairs(image, center, size=20)`**  
  - Draws four small lines centered at `center`  
- **`draw_dashed_line(image, pt1, pt2, dash_length=5, thickness=1)`**  
  - Draws a dashed line segment by segment  
- **`draw_yolo_results(image, detections)`**  
  - Overlays bounding boxes, labels, confidence, and 3D coords  

---

### `robot_ui_module.py`  
**Tkinter UI & Interbotix motion** integration.  
- **`initialize_robot()`**  
  - Connects via ROS 2 to `interbotix_xsarm_control`  
  - Homes the arm to a safe position  
- **`detecting_home_position()`**  
  - Reads and stores current joint/Cartesian pose  
- **`move_robot_to_center(position: Tuple[float, float, float])`**  
  - Moves end effector directly above object center  
- **`generate_smooth_path(corners: List[Tuple[float, float, float]], steps: int=100) -> List[Pose]`**  
  - Interpolates a looped path around the object’s box corners  
- **`move_robot_around_corners(path: List[Pose])`**  
  - Publishes trajectories with non-linear timing for smooth motion  
- **`grip_object_at_position(position: Tuple[float, float, float])`**  
  - Executes a pick sequence: approach, close gripper, lift, and retract  
- **`display_ui(detection_results: List[dict], bot)`**  
  - Renders a Tkinter window listing detections  
  - Provides buttons for **Center**, **Box Loop**, and **Grip** actions  
  - Calls the corresponding motion helper on button press  

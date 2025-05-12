## Module Descriptions

- **main_or_1.py**  
  The Python API entry point. Initializes the RealSense camera and Interbotix robot, then enters the main loop to:
  1. Grab aligned color & depth frames  
  2. Run checkerboard and AprilTag detection  
  3. Run YOLO detection & solvePnP for 3D coordinates  
  4. Update the Tkinter UI with detection results  
  5. Dispatch user-chosen robot motions (center, bounding-box loop, grip)

- **camera_realsense.py**  
  RealSense camera interface class:  
  - Starts depth & color streams and aligns depth to color  
  - `read_frames()` returns `(color_image, depth_frame)` as NumPy arrays  
  - `stop()` stops the pipeline

- **check_intrnsic.py**  
  Standalone script to retrieve and print factory intrinsics for the RealSense color stream at 640×480 resolution.

- **updated_realsense_cam_capture.py**  
  Capture utility for calibration images:  
  - Streams color frames  
  - Press **s** to save a snapshot to a specified directory  
  - Press **q** to quit

- **cam_cal_class.py**  
  Camera calibration script:  
  - Detects 9×6 chessboard corners in saved calibration images  
  - Computes intrinsic matrix (`mtx`), distortion coefficients (`dist`), and extrinsic vectors (`rvecs`, `tvecs`)  
  - Saves parameters to a `.npz` file and prints them  
  - Visualizes pose axes on each image

- **checkerboard_module.py**  
  Checkerboard pose estimation:  
  - Defines the board geometry and square size  
  - `find_checkerboard_pose(color_image, mtx, dist)` locates corners, refines them, and solves PnP to get `rvec` & `tvec`

- **plane_fitting_module.py**  
  Plane-fitting utilities (e.g., with Open3D):  
  - Compute signed distance from 3D points to the fitted checkerboard plane

- **solvepnp_helpers.py**  
  Helpers for solvePnP and visualization:  
  - Convert rotation matrices to Euler angles  
  - Draw coordinate axes or dashed lines on images

- **yolo_module.py**  
  YOLOv8 detection wrapper:  
  - Loads a `best.pt` model  
  - `run_yolo_detections(color_image, depth_frame, mtx, dist, …)` runs inference, filters classes, back-projects 2D→3D, and returns detection dictionaries

- **draw_helpers.py**  
  Drawing utilities for visualization:  
  - `draw_crosshairs(image)` draws four markers on the image  
  - `draw_dashed_line(image, pt1, pt2, …)` draws a dashed line between two points  
  - `draw_yolo_results(color_image, detections)` overlays YOLO bounding boxes, labels, center markers, and world coordinates

- **robot_ui_module.py**  
  Tkinter-based UI and robot motion helpers:  
  - Robot initialization, homing, simple moves (center, box loop, grip)  
  - `display_ui(detection_results, bot)` presents detection list and lets the user choose an action  

You can copy this section directly into your README under a “Module Descriptions” heading. Adjust any function names or details as needed.

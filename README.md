# Handover Hard Part Detection for MAS500

## Overview
This repository implements a ROS 2–based system for detecting and handling “hard” parts in robotic hand-over tasks as part of the MAS500 project. It provides perception and detection , example launch configurations, integration with Interbotix manipulators, and a Python API entry point that ties together modular components.

## Prerequisites
- **Operating System**    : **Ubuntu 22.04.5 LTS** — **native** (bare-metal or dual-boot) installation required  
- **ROS 2 Distribution** : **Humble Hawksbill**  
- **Python**             : ≥ 3.10  
- **pip**  



## Installation
1. **Clone this repository**
   ```bash
   git clone https://github.com/omaralhamza/handover_hard_part_detection_mas500.git
   cd handover_hard_part_detection_mas500
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r scripts_hand_over/requirements.txt
   ```

## Interbotix Manipulator Setup
For complete instructions on installing the Interbotix ROS 2 packages on Ubuntu or Raspberry Pi, see the official docs:  
<https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros_interface/ros2/software_setup.html>

### Quick Install on **Ubuntu + ROS 2 Humble**
```bash
sudo apt install curl
curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d humble
```

## Usage

### update all paths that reference differnt parts such as yolo model path 

yolo_model_path = "/home/omar/handover_hard_part_detection_mas500/scripts_hand_over/best.pt"


### 1  Launch Interbotix Robot
If you are using the **VX300** model:
```bash
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300
```
*To use a different model, replace `vx300` with your robot’s model name.*

> **Note:** Running the full MAS500 project with a different Interbotix robot requires updating **all** references to `vx300` in launch files and scripts.

### 2 Run the Full Project via Python API
```bash
python3 main_or_1.py
```
This initializes camera input, YOLO detection, plane fitting, pose estimation, and the Tkinter-based robot UI.

## Repository Structure
```plaintext
handover_hard_part_detection_mas500/
├── scripts_hand_over/              # ROS 2 nodes, launch files, configs
    ├── runs/detect                 # Training and test evaluation metrics for yolo model
    ├── live_processed_pictures     # Contains 2 folder which save the last 5 raw and yolo taken pictures 
    ├── requirements.txt            # Python dependencies
    ├── main_or_1.py                # Python API entry point (runs full pipeline)
    ├── robot_ui_module.py          # Tkinter UI & robot-motion helpers
    ├── plane_fitting_module.py     # Checkerboard plane-fitting utilities
    ├── solvepnp_helpers.py         # solvePnP & drawing helpers
    ├── yolo_module.py              # YOLO detection wrapper
    ├── camera_realsense.py         # Intel RealSense camera interface
    ├── checkerboard_module.py      # Checkerboard detection & pose estimation
    └── README.md                   # Code documentation
└── README.md                       # Project documentation
Other codes that are included but not needed to run the project such as:
check_intrnsic.py                   # Get the facory intrinsic parameters of the connected camera based on the resolution used.
Use the following codes if you want to run checkerboard calibration isntead of using the factory:
cam_cal_class.py, updated_realsense_cam_capture.py 
draw_helpers.py                      # Mainly not used in the current setup but very helpful for debugging 
```






## Maintainer
**Omar Alhamza** — <omaraalhamza@gmail.com>

## Code overview in the README INSIDE THE scripts_hand_over FOLDER

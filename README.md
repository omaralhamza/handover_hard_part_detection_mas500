# Handover Hard Part Detection for MAS500

## Overview
This repository implements a ROS 2–based system for detecting and handling “hard” parts in robotic handover tasks as part of the MAS500 project. It provides perception and detection nodes, example launch configurations, integration with Interbotix manipulators, and a Python API entry point that ties together modular components.

## Prerequisites
- **Operating System**: Ubuntu 22.04.5 LTS (native installation required)  
- **ROS 2 Distribution**: Humble Hawksbill  
- **Python**: ≥ 3.10  
- **pip**

##Installation
1. Clone this repository

git clone https://github.com/omaralhamza/handover_hard_part_detection_mas500.git
cd handover_hard_part_detection_mas500
Install Python dependencies

bash
Copy
Edit
pip install -r scripts_hand_over/requirements.txt
Interbotix Manipulator Setup
For more information about installing the Interbotix ROS 2 packages on Ubuntu or Raspberry Pi, see:
https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros_interface/ros2/software_setup.html

Quick Install on Ubuntu with ROS 2 Humble
bash
Copy
Edit
sudo apt install curl
curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d humble
Usage
Launch Handover Hard Part Detection
bash
Copy
Edit
ros2 launch scripts_hand_over detection_launch.py
You can pass additional launch arguments to customize topics, parameters, or behavior.

Launch Interbotix Robot
If you are using the VX300 model:

bash
Copy
Edit
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300
To use a different model, replace vx300 with your robot’s model name.

Note: If running the full MAS500 project with a different Interbotix robot, update any references to vx300 in your launch files and scripts to match your robot model.

Run the Full Project via Python API
After the robot is running:

bash
Copy
Edit
python3 main_or_1.py
This initializes camera input, YOLO detection, plane fitting, pose estimation, and the Tkinter-based robot UI.

Repository Structure
plaintext
Copy
Edit
handover_hard_part_detection_mas500/
├── scripts_hand_over/          # ROS 2 nodes, launch files, and configs
│   ├── requirements.txt        # Python dependencies
│   ├── detection_node.py       # ROS 2 node for hard-part detection
│   ├── detection_launch.py     # ROS 2 launch file
│   └── config/                 # Configuration files (YAML, etc.)
├── main_or_1.py                # Python API entry point (runs full pipeline)
├── robot_ui_module.py          # UI and robot movement helpers
├── plane_fitting_module.py     # Checkerboard plane fitting functions
├── solvepnp_helpers.py         # solvePnP and drawing helpers
├── yolo_module.py              # YOLO detection wrapper
├── camera_realsense.py         # RealSense camera interface
├── checkerboard_module.py      # Checkerboard detection and pose estimation
└── README.md                   # Project documentation
Contributing
Contributions are welcome! Please open an issue or submit a pull request for:

Bug fixes

Feature enhancements

Documentation improvements

License
This project is released under the MIT License.

Maintainer
Omar Alhamza (your.email@example.com)

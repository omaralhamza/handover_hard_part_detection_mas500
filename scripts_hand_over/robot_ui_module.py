import tkinter as tk
import threading
import time
import math  # needed for sqrt, etc.
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

selection_active = False

# --------------------------------------------------
# Robot constants & functions
# --------------------------------------------------
def initialize_robot():
    """
    Initializes the Interbotix robot, starts it up, and moves it to the home position.
    """
    bot = InterbotixManipulatorXS(
        robot_model='vx300',
        group_name='arm',
        gripper_name='gripper'
    )
    robot_startup()
    detecting_home_position(bot)
    return bot

def detecting_home_position(bot):
    """
    Moves the robot to a 'home' position for detection and releases the gripper.
    For example, sets x=0.3 and z=0.57 (in meters) then releases.
    """
    bot.arm.set_ee_pose_components(x=0.2, z=0.55)
    time.sleep(1)
    bot.gripper.release()

def move_robot_to_center(bot, center_x_m, center_y_m, center_z_m):
    """
    Moves the robot to the specified center in the robot frame (in meters).
    Uses the given center_z directly.
    """
    target_z = center_z_m 
    print(f"[DEBUG] Moving to center => X={center_x_m:.3f}m, Y={center_y_m:.3f}m, Z={target_z:.3f}m")
    bot.arm.set_ee_pose_components(x=center_x_m, y=center_y_m, z=target_z+0.1)
    print("Returning to detecting position...")
    time.sleep(3)
    detecting_home_position(bot)

# --------------------------------------------------
# Advanced Bounding Box Path Movement
# --------------------------------------------------
TRAVEL_SPEED = 0.5
TIME_SCALE_EXPONENT = 0.35
SHORT_DISTANCE_THRESHOLD = 0.01  # 2 cm
DELAY_REDUCTION_FACTOR = 0.1

def generate_smooth_path(corners, steps=1):
    """
    Given a list of corners in the form (x_m, y_m, z_m),
    subdivides each edge into 'steps+1' segments and closes the loop.
    Returns a new list of points including intermediate interpolations.
    """
    smooth_points = []
    n = len(corners)
    if n < 2:
        return corners

    for i in range(n):
        c_start = corners[i]
        c_end = corners[(i + 1) % n]
        for s in range(steps + 1):
            t = s / float(steps) if steps != 0 else 0
            x_ = c_start[0] + t * (c_end[0] - c_start[0])
            y_ = c_start[1] + t * (c_end[1] - c_start[1])
            z_ = c_start[2] + t * (c_end[2] - c_start[2])
            smooth_points.append((x_, y_, z_))
    return smooth_points

def move_robot_around_corners(bot, corners, base_z, loops=1, steps=1):
    """
    Moves the robot around the given corners in meters,
    using time-based non-linear scaling for short or longer moves.
    The z axis is kept constant based on the provided base_z plus an offset.
    
    Parameters:
      bot    : the robot object
      corners: list of (x, y, z) for the bounding box (used only for x and y)
      base_z : the detected object's center z (in m) from the UI (e.g. 0.216)
      loops  : number of loops (default=1)
      steps  : interpolation steps per edge (default=1)
    
    The constant z is computed as: constant_z = base_z + offset.
    You can change the offset below (currently set to 0.2 m).
    """
    if not corners:
        print("[WARN] No corners to move around.")
        return

    smooth_path = generate_smooth_path(corners, steps=steps)
    
    offset = 0.13  # Change this value if you want a different offset
    constant_z = base_z + offset  # For example, if base_z is 0.216 then constant_z becomes 0.416 m.
    print(f"[DEBUG] Using constant z value: {constant_z:.3f} m (base: {base_z:.3f} + offset: {offset:.3f})")

    for loop in range(loops):
        print(f"[DEBUG] Starting loop {loop + 1} around bounding box.")
        prev_x = prev_y = prev_z = None

        # Iterate over the smooth path using only the x and y coordinates.
        # The original z value is ignored and replaced with constant_z.
        for i, (x_robot, y_robot, _z_robot) in enumerate(smooth_path):
            target_z = constant_z  # Always use the computed constant z.
            if abs(x_robot) > 1.0 or abs(y_robot) > 1.0:
                print(f"[WARN] Skipping out-of-reach corner: X={x_robot:.3f}, Y={y_robot:.3f}")
                continue

            print(f"[DEBUG] - Move {i+1}/{len(smooth_path)} => X={x_robot:.3f}, Y={y_robot:.3f}, Z={target_z:.3f}")
            bot.arm.set_ee_pose_components(x=x_robot, y=y_robot, z=target_z)

            if prev_x is not None:
                distance = math.sqrt((x_robot - prev_x)**2 +
                                     (y_robot - prev_y)**2 +
                                     (target_z - prev_z)**2)
                base_time = distance / TRAVEL_SPEED
                travel_time = base_time if distance < SHORT_DISTANCE_THRESHOLD else base_time ** TIME_SCALE_EXPONENT
            else:
                travel_time = 0

            time.sleep(travel_time * DELAY_REDUCTION_FACTOR)
            prev_x, prev_y, prev_z = x_robot, y_robot, target_z

    print("[DEBUG] Completed bounding box loops. Returning to detecting position.")
    detecting_home_position(bot)

def grip_object_at_position(bot, robot_x, robot_y, robot_z):
    """
    Moves the robot to the object's position and activates the gripper.
    Here, robot_z is assumed to be the object's height above the table.
    Steps:
      1. Compute target approach and grip poses using the object height.
      2. Move to an approach pose (20 cm above the object).
      3. Lower the arm to the grip pose (5 cm above the object).
      4. Command the gripper to grasp the object and wait until the grasp is complete.
      5. Lift the object by moving back to the approach pose.
      6. Return to the home (detecting) position.
      7. Release the gripper after reaching home.
    """
    if robot_z < 0.015:
        object_height = 0.03
    else:
        object_height = robot_z

    approach_z = object_height + 0.2  
    grip_z = object_height + 0.06  


    print(f"[DEBUG] Lowering to grip position at Z={grip_z:.3f}")
    bot.arm.set_ee_pose_components(x=robot_x+0.015, y=robot_y, z=grip_z)
    time.sleep(2)

    print("[DEBUG] Activating gripper to grasp the object")
    bot.gripper.grasp()
    time.sleep(5)

    print("[DEBUG] Lifting the object")
    bot.arm.set_ee_pose_components(x=robot_x, y=robot_y, z=approach_z)
    time.sleep(2)

    print("Returning to detecting position...")
    detecting_home_position(bot)
    time.sleep(2)
    print("[DEBUG] Releasing gripper after reaching home")
    bot.gripper.release()

# --------------------------------------------------
# Tkinter UI for selection
# --------------------------------------------------
def close_menu(root):
    """Helper function to destroy the Tkinter window and reset the flag."""
    global selection_active
    root.destroy()
    selection_active = False

def display_ui(detection_results, bot):
    """
    Shows a Tkinter UI listing robot-frame coordinates (in meters).
    Each detection is a tuple:
      (label, class_name, x_center, y_center, confidence,
       robot_x, robot_y, robot_z, corners_list, robot_info_str)
    """
    global selection_active
    if selection_active:
        return

    selection_active = True
    root = tk.Tk()
    root.title("Select Coordinate and Movement Type for Robot")

    tk.Label(root, text="Detected Robot-Frame Coordinates").pack()
    listbox = tk.Listbox(root, width=100)
    listbox.pack()

    mapping = []
    for i, detection in enumerate(detection_results):
        label, class_name, x_center, y_center, confidence, robot_x, robot_y, robot_z, corners, robot_info = detection
        line = (f"{label} - {class_name} - Conf: {confidence:.2f} - "
                f"Robot Frame: X:{robot_x:.3f}, Y:{robot_y:.3f}, Z:{robot_z:.3f} m")
        listbox.insert(tk.END, line)
        mapping.append(i)

    def on_select():
        selection = listbox.curselection()
        if selection:
            idx = mapping[selection[0]]
            label, class_name, x_center, y_center, confidence, robot_x, robot_y, robot_z, corners, robot_info = detection_results[idx]

            move_window = tk.Toplevel(root)
            move_window.title("Select Movement Type")
            tk.Label(move_window, text="Where would you like to move?").pack(pady=10)

            def move_to_center():
                move_robot_to_center(bot, robot_x + 0.02, robot_y, robot_z)
                move_window.destroy()
                close_menu(root)

            def move_to_bounding_box():
                # Pass the detected center z (robot_z) as base_z.
                move_robot_around_corners(bot, corners, base_z=robot_z, steps=1)
                move_window.destroy()
                close_menu(root)

            def grip_object():
                grip_object_at_position(bot, robot_x, robot_y, robot_z)
                move_window.destroy()
                close_menu(root)

            tk.Button(move_window, text="Center", command=move_to_center, width=15)\
                .pack(side=tk.LEFT, padx=10, pady=10)
            tk.Button(move_window, text="Bounding Box", command=move_to_bounding_box, width=15)\
                .pack(side=tk.LEFT, padx=10, pady=10)
            tk.Button(move_window, text="Grip", command=grip_object, width=15)\
                .pack(side=tk.LEFT, padx=10, pady=10)
        else:
            close_menu(root)

    tk.Button(root, text="Select", command=on_select).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", lambda: close_menu(root))
    root.mainloop()

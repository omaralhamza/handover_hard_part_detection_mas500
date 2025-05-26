import tkinter as tk
import threading
import time
import math
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

selection_active = False


MAX_REACH       = 0.75
SAFETY_FACTOR   = 0.80
SAFETY_LIMIT    = MAX_REACH * SAFETY_FACTOR


def initialize_robot():
    bot = InterbotixManipulatorXS(
        robot_model='vx300',
        group_name='arm',
        gripper_name='gripper'
    )
    robot_startup()
    detecting_home_position(bot)
    return bot


def detecting_home_position(bot):
    bot.arm.set_ee_pose_components(x=0.2, z=0.55)
    time.sleep(1)
    bot.gripper.release()


def move_robot_to_center(bot, center_x_m, center_y_m, center_z_m):
    if abs(center_x_m) > SAFETY_LIMIT or abs(center_y_m) > SAFETY_LIMIT:
        print(f"[WARN] Center target out of safety bounds: X={center_x_m:.3f}, Y={center_y_m:.3f} (limit ±{SAFETY_LIMIT:.2f} m)")
        return
    target_z = center_z_m
    print(f"[DEBUG] Moving to center => X={center_x_m:.3f}m, Y={center_y_m:.3f}m, Z={target_z:.3f}m")
    bot.arm.set_ee_pose_components(x=center_x_m, y=center_y_m, z=target_z+0.1)
    print("Returning to detecting position...")
    time.sleep(3)
    detecting_home_position(bot)


TRAVEL_SPEED = 0.5
TIME_SCALE_EXPONENT = 0.35
SHORT_DISTANCE_THRESHOLD = 0.01
DELAY_REDUCTION_FACTOR = 0.1

def generate_smooth_path(corners, steps=1):
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
    if not corners:
        print("[WARN] No corners to move around.")
        return
    for x, y, _ in corners:
        if abs(x) > SAFETY_LIMIT or abs(y) > SAFETY_LIMIT:
            print(f"[WARN] Bounding-box move skipped: corner out of safety bounds (X={x:.3f}, Y={y:.3f}, limit ±{SAFETY_LIMIT:.2f} m)")
            return
    smooth_path = generate_smooth_path(corners, steps=steps)
    offset = 0.13
    constant_z = base_z + offset
    print(f"[DEBUG] Using constant z value: {constant_z:.3f} m (base: {base_z:.3f} + offset: {offset:.3f})")
    for loop in range(loops):
        print(f"[DEBUG] Starting loop {loop + 1} around bounding box.")
        prev_x = prev_y = prev_z = None
        for i, (x_robot, y_robot, _z_robot) in enumerate(smooth_path):
            target_z = constant_z
            print(f"[DEBUG] - Move {i+1}/{len(smooth_path)} => X={x_robot:.3f}, Y={y_robot:.3f}, Z={target_z:.3f}")
            bot.arm.set_ee_pose_components(x=x_robot, y=y_robot, z=target_z)
            if prev_x is not None:
                distance = math.sqrt((x_robot - prev_x)**2 + (y_robot - prev_y)**2 + (target_z - prev_z)**2)
                base_time = distance / TRAVEL_SPEED
                travel_time = base_time if distance < SHORT_DISTANCE_THRESHOLD else base_time ** TIME_SCALE_EXPONENT
            else:
                travel_time = 0
            time.sleep(travel_time * DELAY_REDUCTION_FACTOR)
            prev_x, prev_y, prev_z = x_robot, y_robot, target_z
    print("[DEBUG] Completed bounding box loops. Returning to detecting position.")
    detecting_home_position(bot)


def grip_object_at_position(bot, robot_x, robot_y, robot_z):
    angle = math.atan2(robot_y, robot_x)
    offset = 0.015  
    adj_x = robot_x + offset * math.cos(angle)
    adj_y = robot_y + offset * math.sin(angle)
    if abs(adj_x) > SAFETY_LIMIT or abs(adj_y) > SAFETY_LIMIT:
        print(f"[WARN] Grip target out of safety bounds after offset: X={adj_x:.3f}, Y={adj_y:.3f} (limit ±{SAFETY_LIMIT:.2f} m)")
        return
    if robot_z < 0.015:
        object_height = 0.03
    else:
        object_height = robot_z
    approach_z = object_height + 0.2
    grip_z = object_height + 0.061
    print(f"[DEBUG] Lowering to grip position at adjusted (X={adj_x:.3f}, Y={adj_y:.3f}, Z={grip_z:.3f})")
    bot.arm.set_ee_pose_components(x=adj_x, y=adj_y, z=grip_z)
    time.sleep(2)
    print("[DEBUG] Activating gripper to grasp the object")
    bot.gripper.grasp()
    time.sleep(5)
    print("[DEBUG] Lifting the object")
    bot.arm.set_ee_pose_components(x=adj_x, y=adj_y, z=approach_z)
    time.sleep(2)
    print("Returning to detecting position...")
    detecting_home_position(bot)
    time.sleep(2)
    print("[DEBUG] Releasing gripper after reaching home")
    bot.gripper.release()

def close_menu(root):
    global selection_active
    root.destroy()
    selection_active = False


def display_ui(detection_results, bot):
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
                move_robot_to_center(bot, robot_x, robot_y, robot_z)
                move_window.destroy()
                close_menu(root)
            def move_to_bounding_box():
                move_robot_around_corners(bot, corners, base_z=robot_z, steps=1)
                move_window.destroy()
                close_menu(root)
            def grip_object():
                grip_object_at_position(bot, robot_x, robot_y, robot_z)
                move_window.destroy()
                close_menu(root)
            tk.Button(move_window, text="Center", command=move_to_center, width=15).pack(side=tk.LEFT, padx=10, pady=10)
            tk.Button(move_window, text="Bounding Box", command=move_to_bounding_box, width=15).pack(side=tk.LEFT, padx=10, pady=10)
            tk.Button(move_window, text="Grip", command=grip_object, width=15).pack(side=tk.LEFT, padx=10, pady=10)
        else:
            close_menu(root)
    tk.Button(root, text="Select", command=on_select).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", lambda: close_menu(root))
    root.mainloop()

from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Path to your trained YOLO model
yolo_model_path = "/home/omar/Cameras/best.pt"
yolo_model = YOLO(yolo_model_path)

# Classes to ignore
ignore_classes = {"jacket", "shirt"}

# Counter for labeling classes (#class_1, #class_2, etc.)
class_counts = defaultdict(int)

def run_yolo_detections(color_image, depth_frame, mtx, dist, last_R, last_tvec):
    """
    Runs YOLO on 'color_image', returning a list of dicts:
      {
        "hashtag": "#class_#",   # e.g. #person_1
        "class_name": class_str, # e.g. 'person'
        "box": (x1,y1,x2,y2),
        "conf": float,
        "center": (cx, cy),
        "world": (Xw_mm, Yw_mm, Zw_mm),
        "camera_m": (Xc_m, Yc_m, Zc_m),
        "corners_robot": [ ... ] # (filled later if you wish)
      }
    """
    class_counts.clear()
    results = yolo_model(color_image, verbose=False)
    detection_list = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names.get(cls_id, f"class_{cls_id}")

            # Skip ignored classes
            if class_name.lower() in ignore_classes:
                continue

            # Convert coords to int
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx_box = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2

            # Calculate depth from the depth frame
            Z = depth_frame.get_distance(cx_box, cy_box)
            Xc, Yc, Zc = (None, None, None)
            Xw, Yw, Zw = (None, None, None)

            if Z > 0:
                # Camera coordinates in meters
                Xc = (cx_box - mtx[0, 2]) * Z / mtx[0, 0]
                Yc = (cy_box - mtx[1, 2]) * Z / mtx[1, 1]
                Zc = Z

                # World coordinates in mm (if we have solvePnP data)
                if last_R is not None and last_tvec is not None:
                    R_inv = np.linalg.inv(last_R)
                    p_cam = np.array([[Xc], [Yc], [Zc]], dtype=np.float32)
                    # last_tvec is in mm, so convert to meters:
                    tvec_meters = last_tvec / 1000.0
                    p_world = R_inv @ (p_cam - tvec_meters)
                    Xw = p_world[0][0] * 1000.0
                    Yw = p_world[1][0] * 1000.0
                    Zw = p_world[2][0] * 1000.0
                else:
                    # fallback: treat camera coords as "world" directly
                    Xw = Xc * 1000.0
                    Yw = Yc * 1000.0
                    Zw = Zc * 1000.0

            class_counts[class_name] += 1
            hashtag = f"#{class_name}_{class_counts[class_name]}"

            detection_list.append({
                "hashtag": hashtag,
                "class_name": class_name,
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "center": (cx_box, cy_box),
                "world": (Xw, Yw, Zw),
                "camera_m": (Xc, Yc, Zc),
                "corners_robot": []  # can fill in later
            })

    return detection_list
    
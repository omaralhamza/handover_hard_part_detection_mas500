from ultralytics import YOLO
import numpy as np
from collections import defaultdict

yolo_model_path = "/home/omar/handover_hard_part_detection_mas500/scripts_hand_over/best.pt"
yolo_model = YOLO(yolo_model_path)

class_counts = defaultdict(int)

def run_yolo_detections(color_image, depth_frame, mtx, dist, last_R, last_tvec):
    class_counts.clear()
    results = yolo_model(color_image, verbose=False)
    detection_list = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx_box = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2
            Z = depth_frame.get_distance(cx_box, cy_box)
            Xc = Yc = Zc = None
            Xw = Yw = Zw = None
            if Z > 0:
                Xc = (cx_box - mtx[0, 2]) * Z / mtx[0, 0]
                Yc = (cy_box - mtx[1, 2]) * Z / mtx[1, 1]
                Zc = Z
                if last_R is not None and last_tvec is not None:
                    R_inv = np.linalg.inv(last_R)
                    p_cam = np.array([[Xc], [Yc], [Zc]], dtype=np.float32)
                    tvec_meters = last_tvec / 1000.0
                    p_world = R_inv @ (p_cam - tvec_meters)
                    Xw, Yw, Zw = (p_world[:, 0] * 1000.0).tolist()
                else:
                    Xw, Yw, Zw = Xc * 1000.0, Yc * 1000.0, Zc * 1000.0
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
                "corners_robot": []
            })
    return detection_list


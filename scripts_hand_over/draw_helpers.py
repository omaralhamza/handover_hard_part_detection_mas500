import cv2
import numpy as np

def draw_crosshairs(image):
    h, w = image.shape[:2]
    cx = w // 2
    cy = h // 2

    # Crosshair 0
    cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
    # Crosshair 1
    cx1, cy1 = cx + 500, cy
    cv2.drawMarker(image, (cx1, cy1), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
    # Crosshair 2
    cx2, cy2 = cx + 500, cy - 250
    cv2.drawMarker(image, (cx2, cy2), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
    # Crosshair 3
    cx3, cy3 = cx, cy - 250
    cv2.drawMarker(image, (cx3, cy3), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)

    return cx, cy, cx3, cy3

def draw_dashed_line(image, pt1, pt2, color=(255,105,180), thickness=2, num_dashes=15):
    dash_length = 10
    gap_length  = 10
    (x1, y1), (x2, y2) = pt1, pt2
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    dash_gap_total = dash_length + gap_length
    dash_count = int(dist / dash_gap_total)
    for i in range(dash_count):
        t1 = i / dash_count
        t2 = (i + 0.5) / dash_count
        xi1 = int(x1*(1 - t1) + x2*t1)
        yi1 = int(y1*(1 - t1) + y2*t1)
        xi2 = int(x1*(1 - t2) + x2*t2)
        yi2 = int(y1*(1 - t2) + y2*t2)
        cv2.line(image, (xi1, yi1), (xi2, yi2), color, thickness)

def draw_yolo_results(color_image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        hashtag = det["hashtag"]
        conf = det["conf"]
        (Xw, Yw, Zw) = det["world"]
        cx_box, cy_box = det["center"]

        # Draw bounding box
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
        label_text = f"{hashtag}: {conf:.2f}"
        cv2.putText(color_image, label_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.drawMarker(color_image, (cx_box, cy_box), (0,255,0),
                       cv2.MARKER_CROSS, 10, 2)
        if Xw is not None:
            coords_text = f"Checkerboard: X:{Xw:.1f} mm, Y:{Yw:.1f} mm, Z:{Zw:.1f} mm"
            cv2.putText(color_image, coords_text, (x1, y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

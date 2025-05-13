#!/usr/bin/env python3
"""
visualize_bboxes.py

Quickly preview YOLO-format bounding boxes on images of arbitrary size.
Scales large images down so the longest side ≤ maxside, draws boxes
(color-coded by class id), and shows each window for `wait` seconds
(or until you press 'q').

Usage:
    python visualize_bboxes.py \
        --imgdir /path/to/images \
        --lbldir /path/to/labels \
        [--maxside 1280] [--wait 1.0] [--sample 100]
"""

import cv2
import argparse
import random
from pathlib import Path

def yolo_to_pixel(box, W, H):
    """Convert YOLO [xc, yc, bw, bh] normalized coords to pixel coords."""
    xc, yc, bw, bh = map(float, box)
    x1 = int((xc - bw/2) * W)
    y1 = int((yc - bh/2) * H)
    x2 = int((xc + bw/2) * W)
    y2 = int((yc + bh/2) * H)
    return x1, y1, x2, y2

def get_color(class_id: int):
    """Deterministic pseudo-color map for class ids."""
    # simple hashing to get reproducible but distinct colors
    return ((class_id * 37) % 255,
            (class_id * 17) % 255,
            (class_id * 29) % 255)

def visualize(img_dir: Path, lbl_dir: Path, maxside: int, wait: float, sample: int):
    # collect image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    if sample and sample < len(imgs):
        imgs = random.sample(imgs, sample)
    if not imgs:
        print(f"[!] No images found in {img_dir}")
        return

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[!] Could not read {img_path}")
            continue

        H, W = img.shape[:2]
        # scale factor to fit maxside
        scale = min(1.0, maxside / max(H, W))
        if scale < 1.0:
            disp = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = img.copy()

        # overlay boxes
        lbl_file = lbl_dir / f"{img_path.stem}.txt"
        if lbl_file.exists():
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cid, *box = parts
                x1, y1, x2, y2 = yolo_to_pixel(box, W, H)
                # apply same scale
                x1, y1, x2, y2 = map(lambda v: int(v * scale), (x1, y1, x2, y2))
                color = get_color(int(cid))
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(disp, cid, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(disp, "NO LABEL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        # show
        cv2.imshow("BBox Preview (q to quit)", disp)
        key = cv2.waitKey(int(wait * 1000)) & 0xFF
        cv2.destroyWindow("BBox Preview (q to quit)")
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO bboxes")
    parser.add_argument("--imgdir",   type=Path, required=True, help="Path to images folder")
    parser.add_argument("--lbldir",   type=Path, required=True, help="Path to labels folder")
    parser.add_argument("--maxside",  type=int,   default=1280, help="Max side length for display")
    parser.add_argument("--wait",     type=float, default=1.0,  help="Seconds to show each image")
    parser.add_argument("--sample",   type=int,   default=0,    help="Show a random subset of N images")
    args = parser.parse_args()

    if not args.imgdir.exists() or not args.lbldir.exists():
        print("[!] imgdir or lbldir does not exist")
        return

    visualize(args.imgdir, args.lbldir, args.maxside, args.wait, args.sample)

if __name__ == "__main__":
    main()


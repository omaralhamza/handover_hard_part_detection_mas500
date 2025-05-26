#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/home/omar/Downloads/NEW_YOLO_TEST_MAS_500")

LABEL_DIR = BASE / "labels"
IMG_DIR   = BASE / "images"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

deleted_labels = []
deleted_images = []

for txt_file in LABEL_DIR.glob("*.txt"):
    content = txt_file.read_text().strip()
    if not content:
        deleted_labels.append(txt_file.name)
        txt_file.unlink()
        stem = txt_file.stem
        for ext in IMG_EXTS:
            img_path = IMG_DIR / f"{stem}{ext}"
            if img_path.exists():
                deleted_images.append(img_path.name)
                img_path.unlink()

print("Deleted label files:")
for name in deleted_labels:
    print(f"  • {name}")

print("\nDeleted image files:")
for name in deleted_images:
    print(f"  • {name}")

print("\nCleanup complete.")

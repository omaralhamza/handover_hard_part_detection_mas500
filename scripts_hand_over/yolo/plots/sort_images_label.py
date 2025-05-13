#!/usr/bin/env python3
from pathlib import Path

# Base directory containing 'labels' and 'images' folders
BASE = Path("/home/omar/Downloads/NEW_YOLO_TEST_MAS_500")

LABEL_DIR = BASE / "labels"
IMG_DIR   = BASE / "images"

# Supported image extensions to check for corresponding images
IMG_EXTS = {".jpg", ".jpeg", ".png"}

deleted_labels = []
deleted_images = []

# Iterate over all .txt files in the labels directory
for txt_file in LABEL_DIR.glob("*.txt"):
    # Read the file content and strip whitespace
    content = txt_file.read_text().strip()
    
    # If the file is empty (no annotation lines), delete it and its corresponding image
    if not content:
        # Record and delete the empty label file
        deleted_labels.append(txt_file.name)
        txt_file.unlink()
        
        # Determine the base name (without extension)
        stem = txt_file.stem
        
        # Look for and delete matching images
        for ext in IMG_EXTS:
            img_path = IMG_DIR / f"{stem}{ext}"
            if img_path.exists():
                deleted_images.append(img_path.name)
                img_path.unlink()

# Print a summary of deleted files
print("Deleted label files:")
for name in deleted_labels:
    print(f"  • {name}")

print("\nDeleted image files:")
for name in deleted_images:
    print(f"  • {name}")

print("\nCleanup complete.")

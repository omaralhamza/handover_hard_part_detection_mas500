#!/usr/bin/env python3
import random
import shutil
from pathlib import Path

# 1) Define your YOLO base directory
BASE_DIR = Path("/home/omar/Downloads/NEW_YOLO_TEST_MAS_500")

# 2) Source folders containing all images and labels
SRC_IMG_DIR = BASE_DIR / "images"
SRC_LBL_DIR = BASE_DIR / "labels"

# 3) Define splits and ratios
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

# 4) Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# 5) Gather all stems that have both an image and a label
stems = []
for img_path in SRC_IMG_DIR.iterdir():
    if img_path.suffix.lower() in IMG_EXTS:
        stem = img_path.stem
        if (SRC_LBL_DIR / f"{stem}.txt").exists():
            stems.append(stem)

# 6) Shuffle for randomness
random.seed(42)
random.shuffle(stems)

# 7) Compute split indices
n = len(stems)
n_train = int(SPLITS["train"] * n)
n_val   = int(SPLITS["val"] * n)
# ensure test gets the remainder
splits = {
    "train": stems[:n_train],
    "val":   stems[n_train:n_train + n_val],
    "test":  stems[n_train + n_val:]
}

# 8) Create directory structure and copy files
for split, names in splits.items():
    img_out_dir = BASE_DIR / split / "images"
    lbl_out_dir = BASE_DIR / split / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    for stem in names:
        # Copy image
        for ext in IMG_EXTS:
            src_img = SRC_IMG_DIR / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, img_out_dir / src_img.name)
                break
        # Copy label
        src_lbl = SRC_LBL_DIR / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, lbl_out_dir / src_lbl.name)

# 9) Print summary
print("Dataset split and copy complete:")
for split, names in splits.items():
    print(f"  {split}: {len(names)} samples")

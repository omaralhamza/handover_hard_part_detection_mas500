#!/usr/bin/env python3
import shutil
from pathlib import Path

# 1. Define your base and target dirs
BASE = Path("/home/omar/Downloads/NEW_YOLO_TEST_MAS_500")
IMG_OUT = BASE / "images"
LBL_OUT = BASE / "labels"

# 2. Create output directories if missing
for d in (IMG_OUT, LBL_OUT):
    d.mkdir(parents=True, exist_ok=True)

# 3. Iterate through each obj_train_data folder
for sub in BASE.iterdir():
    if not (sub.is_dir() and sub.name.endswith("y")):
        continue
    data_dir = sub / "obj_train_data"
    if not data_dir.is_dir():
        continue

    # 4. Move files based on extension
    for f in data_dir.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            target = IMG_OUT / f.name
        elif ext == ".txt":
            target = LBL_OUT / f.name
        else:
            continue  # skip anything else

        # perform move (or use copy2 for metadata preservation)
        shutil.move(str(f), str(target))
        print(f"Moved {f.name} → {target}")

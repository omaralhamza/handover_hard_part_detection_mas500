#!/usr/bin/env python3
import shutil
from pathlib import Path

BASE = Path("/home/omar/Downloads/NEW_YOLO_TEST_MAS_500")
IMG_OUT = BASE / "images"
LBL_OUT = BASE / "labels"

for d in (IMG_OUT, LBL_OUT):
    d.mkdir(parents=True, exist_ok=True)

for sub in BASE.iterdir():
    if not (sub.is_dir() and sub.name.endswith("y")):
        continue
    data_dir = sub / "obj_train_data"
    if not data_dir.is_dir():
        continue

    for f in data_dir.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            target = IMG_OUT / f.name
        elif ext == ".txt":
            target = LBL_OUT / f.name
        else:
            continue

        shutil.move(str(f), str(target))
        print(f"Moved {f.name} → {target}")

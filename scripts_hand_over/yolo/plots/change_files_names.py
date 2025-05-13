#!/usr/bin/env python3
import pathlib

BASE = pathlib.Path("/home/omar/Downloads/NEW_YOLO_TEST_MAS_500")

# Iterate all subdirectories ending with 'y'
for sub in BASE.iterdir():
    if not (sub.is_dir() and sub.name.endswith("y")):
        continue

    data_dir = sub / "obj_train_data"
    if not data_dir.is_dir():
        continue

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        # Skip if already has the prefix
        if file.name.startswith(sub.name + "_"):
            continue

        # Only rename typical label/image extensions (optional)
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".txt"}:
            continue

        new_name = f"{sub.name}_{file.name}"
        file.rename(data_dir / new_name)
        print(f"Renamed {file.name} → {new_name}")

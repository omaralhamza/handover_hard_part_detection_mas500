#!/usr/bin/env python3
import pathlib

# 1. Point this at your folder of mixed images+labels
LABEL_DIR = pathlib.Path('/home/omar/Downloads/NEW_YOLO_TEST_MAS_500/1174738y/obj_train_data/')

# 2. Iterate over all .txt files only
for txt_path in LABEL_DIR.rglob('*.txt'):
    lines_out = []
    with txt_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                # skip malformed lines
                continue
            # invert 0<->1, leave any other id unchanged
            if cid in (0, 1):
                parts[0] = str(1 - cid)
            lines_out.append(' '.join(parts))

    # 3. Overwrite with flipped IDs
    with txt_path.open('w') as f:
        f.write('\n'.join(lines_out) + '\n')

    print(f'Fixed: {txt_path}')

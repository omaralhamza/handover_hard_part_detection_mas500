import pandas as pd
import matplotlib.pyplot as plt

# Path to your mixed-data results.csv
csv_path = '/home/omar/Downloads/NEW_YOLO_TEST_MAS_500/yolo/runs/detect/train6/results.csv'

# 1. Load your metrics CSV
df = pd.read_csv(csv_path)

# 2. Compute a total loss column for train and val
df['train/total_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
df['val/total_loss']   = df['val/box_loss']   + df['val/cls_loss']   + df['val/dfl_loss']

# 3. Set up a 2×2 grid of subplots for individual losses
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# Add a big figure-level title
fig.suptitle('YOLO Training & Validation Losses (Mixed Data)', fontsize=16)
# Move subplots down to make room for the suptitle
fig.subplots_adjust(top=0.90)

# Top-left: Total Loss
axes[0, 0].plot(df['epoch'], df['train/total_loss'], label='train total')
axes[0, 0].plot(df['epoch'], df['val/total_loss'],   label='val total')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Top-right: Box Loss
axes[0, 1].plot(df['epoch'], df['train/box_loss'], label='train box')
axes[0, 1].plot(df['epoch'], df['val/box_loss'],   label='val box')
axes[0, 1].set_title('Box Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Bottom-left: Classification Loss
axes[1, 0].plot(df['epoch'], df['train/cls_loss'], label='train cls')
axes[1, 0].plot(df['epoch'], df['val/cls_loss'],   label='val cls')
axes[1, 0].set_title('Classification Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Bottom-right: Distribution Focal Loss
axes[1, 1].plot(df['epoch'], df['train/dfl_loss'], label='train dfl')
axes[1, 1].plot(df['epoch'], df['val/dfl_loss'],   label='val dfl')
axes[1, 1].set_title('Distribution Focal Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# 4. Separate plot for Total Loss only
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['train/total_loss'], label='train total')
plt.plot(df['epoch'], df['val/total_loss'],   label='val total')
plt.title('Total Loss Over Epochs (Mixed Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

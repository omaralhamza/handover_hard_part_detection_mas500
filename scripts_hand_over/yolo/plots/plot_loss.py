import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('runs/detect/train6/results.csv')

df['train/total_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
df['val/total_loss']   = df['val/box_loss']   + df['val/cls_loss']   + df['val/dfl_loss']

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(df['epoch'], df['train/total_loss'], label='train total')
axes[0, 0].plot(df['epoch'], df['val/total_loss'],   label='val total')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(df['epoch'], df['train/box_loss'], label='train box')
axes[0, 1].plot(df['epoch'], df['val/box_loss'],   label='val box')
axes[0, 1].set_title('Box Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(df['epoch'], df['train/cls_loss'], label='train cls')
axes[1, 0].plot(df['epoch'], df['val/cls_loss'],   label='val cls')
axes[1, 0].set_title('Classification Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(df['epoch'], df['train/dfl_loss'], label='train dfl')
axes[1, 1].plot(df['epoch'], df['val/dfl_loss'],   label='val dfl')
axes[1, 1].set_title('Distribution Focal Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

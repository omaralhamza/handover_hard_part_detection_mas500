import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppose `cm_norm` is your full normalized confusion matrix (3×3):
cm_norm = np.array([
    [0.95, 0.05, 0.00],
    [0.10, 0.90, 0.00],
    [0.46, 0.54, 0.00]

])

# 1) Remove index 2 (background) from both axes
cm_reduced = np.delete(np.delete(cm_norm, 2, axis=0), 2, axis=1)

# 2) Define your two remaining class names
labels = ['Zip', 'Buttons']

# 3) Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm_reduced, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Normalized Confusion Matrix (Zip vs. Buttons)")
plt.tight_layout()
plt.show()

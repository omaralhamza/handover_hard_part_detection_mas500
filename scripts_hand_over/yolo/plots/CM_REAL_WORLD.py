import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Your full normalized confusion matrix (3×3):
cm_norm = np.array([
    [0.80, 0.20, 0.00],
    [ 0.12,0.88, 0.00],
    [0.81, 0.19, 0.00]
])

# 1) Remove the 3rd row & column (background; index=2)
cm2 = np.delete(np.delete(cm_norm, 2, axis=0), 2, axis=1)

# 2) Define remaining class labels
labels = ['Zip', 'Button']

# 3) Plot the 2×2 confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm2, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Normalized Confusion Matrix (Zip vs. Button)")
plt.tight_layout()
plt.show()

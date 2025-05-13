import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Plug in your normalized confusion matrix (3×3):
cm_norm = np.array([
    [0.97, 0.03, 0.40],  # Predicted Zip vs [True Zip, True Button, True background]
    [0.05, 0.95, 0.60],  # Predicted Button vs …
    [0.00, 0.00, 0.00]   # Predicted background vs …
])

# 2) Remove row & col index 2 (background)
cm2 = np.delete(np.delete(cm_norm, 2, axis=0), 2, axis=1)

# 3) Define your two classes
labels = ['Zip', 'Button']

# 4) Plot the reduced 2×2 matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm2, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Normalized Confusion Matrix (Zip vs. Button)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def relu(x):
    return np.maximum(0, x)

x = np.linspace(0, 5, 500)
t1 = 1.5
t2 = 3.5

# y1: ReLU 1
y1 = relu(x - t1)
# y2: ReLU 2 (Shifted)
y2 = relu(x - t2)
# y_comb: Difference (Soft Step / Ridge)
# Logic: ReLU(x-t1) - ReLU(x-t2)
# For x < t1: 0 - 0 = 0
# For t1 < x < t2: (x-t1) - 0 = Rising
# For x > t2: (x-t1) - (x-t2) = t2-t1 (Constant)
y_comb = y1 - y2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: ReLU 1
axes[0].plot(x, y1, color='#6C8EBF', linewidth=3)
axes[0].fill_between(x, y1, alpha=0.2, color='#6C8EBF')
axes[0].set_title(r"$h_1 = \text{ReLU}(x - t_1)$", fontsize=16)
axes[0].set_ylim(-0.5, 3.0)
axes[0].axvline(t1, linestyle='--', color='gray', alpha=0.5)

# Plot 2: ReLU 2
axes[1].plot(x, y2, color='#B85450', linewidth=3)
axes[1].fill_between(x, y2, alpha=0.2, color='#B85450')
axes[1].set_title(r"$h_2 = \text{ReLU}(x - t_2)$", fontsize=16)
axes[1].set_ylim(-0.5, 3.0)
axes[1].axvline(t2, linestyle='--', color='gray', alpha=0.5)

# Plot 3: Difference
axes[2].plot(x, y_comb, color='#9673A6', linewidth=3)
axes[2].fill_between(x, y_comb, alpha=0.2, color='#9673A6')
axes[2].set_title(r"$y = h_1 - h_2$ (Soft Step)", fontsize=16)
axes[2].set_ylim(-0.5, 3.0)
axes[2].axvline(t1, linestyle='--', color='gray', alpha=0.5)
axes[2].axvline(t2, linestyle='--', color='gray', alpha=0.5)
axes[2].text(2.5, 1.0, "Constructed Region", ha='center', va='bottom', fontsize=12, color='#666666')

plt.tight_layout()
plt.savefig('chapter_01/images/universal_approximation_relu.png', dpi=300, bbox_inches='tight')

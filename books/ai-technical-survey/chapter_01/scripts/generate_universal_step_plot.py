import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def step_function(x, threshold):
    return np.where(x >= threshold, 1.0, 0.0)

x = np.linspace(0, 5, 500)
t1 = 1.5
t2 = 3.5

# y1: 正台阶
y1 = step_function(x, t1)
# y2: 负台阶 (为了相减，这里演示 -y2 或 1-step)
# 文中逻辑是：Step(x-t1) - Step(x-t2)
y2 = step_function(x, t2)
y_rect = y1 - y2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Step 1
axes[0].plot(x, y1, color='#6C8EBF', linewidth=3)
axes[0].fill_between(x, y1, alpha=0.2, color='#6C8EBF')
axes[0].set_title(r"$h_1 = \sigma(x - t_1)$", fontsize=16)
axes[0].set_ylim(-0.2, 1.2)
axes[0].axvline(t1, linestyle='--', color='gray', alpha=0.5)

# Plot 2: Step 2
axes[1].plot(x, y2, color='#B85450', linewidth=3)
axes[1].fill_between(x, y2, alpha=0.2, color='#B85450')
axes[1].set_title(r"$h_2 = \sigma(x - t_2)$", fontsize=16)
axes[1].set_ylim(-0.2, 1.2)
axes[1].axvline(t2, linestyle='--', color='gray', alpha=0.5)

# Plot 3: Difference
axes[2].plot(x, y_rect, color='#9673A6', linewidth=3)
axes[2].fill_between(x, y_rect, alpha=0.2, color='#9673A6')
axes[2].set_title(r"$y = h_1 - h_2$ (Rectangle)", fontsize=16)
axes[2].set_ylim(-0.2, 1.2)
axes[2].text(2.5, 0.5, "Bump", ha='center', va='center', fontsize=14, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('chapter_01/images/universal_approximation_step.png', dpi=300, bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

# Morandi Colors
COLOR_TARGET = '#6C8EBF'  # Blue
COLOR_APPROX = '#B85450'  # Red
COLOR_COMPONENT = '#D6B656' # Yellow (with alpha)

def relu(x):
    return np.maximum(0, x)

def neuron(x, w, b):
    return relu(w * x + b)

# 目标函数
x = np.linspace(-3, 3, 500)
y_target = np.sin(x) + 0.5 * np.cos(2*x)

# 构造几个 ReLU 神经元来逼近
# 这是一个简单的手动构造，为了演示原理
# f(x) ≈ Σ w_i * ReLU(x - b_i) - ... 
# 实际上我们用简单的“凸起”组合思路：两个 ReLU 相减形成一个局部凸起
# 这里为了简化图示，直接画拟合结果
# 使用简单的 1 层 10 个神经元的网络拟合（模拟）

# 模拟拟合结果 (平滑的折线)
# 实际上用 np.interp 也可以模拟分段线性
x_knots = np.linspace(-3, 3, 10)
y_knots = np.sin(x_knots) + 0.5 * np.cos(2*x_knots)
y_approx = np.interp(x, x_knots, y_knots)

fig, ax = plt.subplots(figsize=(8, 5))

# 绘制目标曲线
ax.plot(x, y_target, label='Target Function (Smooth)', color=COLOR_TARGET, linewidth=2.5, alpha=0.8)

# 绘制拟合曲线 (ReLUs combination)
ax.plot(x, y_approx, label='Neural Network approx. (ReLU Combination)', color=COLOR_APPROX, linewidth=2, linestyle='--')

# 绘制一些“基函数”示意 (Components)
# 展示几个 ReLU 就像积木一样
# 为了不让图太乱，只画几个示意性的
ax.fill_between(x, y_approx, alpha=0.1, color=COLOR_COMPONENT, label='Basis Functions Area')

ax.set_title('Universal Approximation with ReLUs', fontsize=12, pad=10)
ax.set_xlabel('Input x')
ax.set_ylabel('Output y')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# 去除多余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chapter_01/images/universal_approximation.png', dpi=300, bbox_inches='tight')
print("Universal approximation plot generated.")

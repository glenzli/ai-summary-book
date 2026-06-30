import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def plot_perceptron_geometry():
    # Setup styling
    # utils.plot_style is automatically applied on import
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Generate linearly separable data
    np.random.seed(42)
    # Class 1 (Positive)
    X1 = np.random.randn(20, 2) * 0.5 + [2, 2]
    # Class 2 (Negative)
    X2 = np.random.randn(20, 2) * 0.5 + [-1, -1]
    
    # 2. Define decision boundary: w1*x1 + w2*x2 + b = 0
    # Let w = [1, 1], b = -1
    # Line equation: x2 = (-w1/w2)*x1 - (b/w2)
    w = np.array([1, 1])
    b = -1.0
    
    x_range = np.linspace(-3, 4, 100)
    y_decision = (-w[0] * x_range - b) / w[1]
    
    # 3. Plot data points
    ax.scatter(X1[:, 0], X1[:, 1], c='#82B366', marker='o', s=100, label='Class +1', edgecolors='white')
    ax.scatter(X2[:, 0], X2[:, 1], c='#B85450', marker='s', s=100, label='Class -1', edgecolors='white')
    
    # 4. Plot decision boundary
    ax.plot(x_range, y_decision, color='#6C8EBF', linewidth=3, label='Decision Boundary\n$w^T x + b = 0$')
    
    # 5. Plot normal vector w
    # Choose a point on the line as origin for vector
    origin = np.array([1.0, (-w[0]*1.0 - b)/w[1]]) # Point on line where x=1.0
    # Scale w for visualization
    w_vis = w * 0.5
    ax.arrow(origin[0], origin[1], w_vis[0], w_vis[1], head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.text(origin[0] + w_vis[0] + 0.1, origin[1] + w_vis[1], r'$\mathbf{w}$', fontsize=14, fontweight='bold')
    
    # 6. Fill regions
    # Above line (wTx + b > 0)
    ax.fill_between(x_range, y_decision, 4, color='#82B366', alpha=0.1)
    # Below line (wTx + b < 0)
    ax.fill_between(x_range, -3, y_decision, color='#B85450', alpha=0.1)

    # 7. Annotations
    ax.text(3, 3, '$y = +1$', fontsize=16, color='#666666', ha='center')
    ax.text(-2, -2, '$y = -1$', fontsize=16, color='#666666', ha='center')
    
    # Styling
    ax.set_title('Geometric Interpretation of Perceptron', fontsize=16, pad=20)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(-3, 4)
    ax.set_ylim(-3, 4)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Equation annotation
    eq_text = r"$\mathbf{w}^T \mathbf{x} + b > 0 \Rightarrow +1$" + "\n" + r"$\mathbf{w}^T \mathbf{x} + b < 0 \Rightarrow -1$"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.05, eq_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=props)

    output_path = 'chapter_01/images/perceptron_geometry.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_perceptron_geometry()

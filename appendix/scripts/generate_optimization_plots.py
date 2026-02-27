import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_negative_gradient_1d():
    """
    Plot a 1D function f(x) = x^2, showing the derivative and the negative gradient step.
    """
    x = np.linspace(-2, 2, 100)
    y = x**2
    
    x0 = 1.5
    y0 = x0**2
    grad = 2 * x0 # derivative at x0
    
    # Tangent line: y - y0 = m(x - x0) => y = m(x - x0) + y0
    tangent_x = np.linspace(0.5, 2.5, 10)
    tangent_y = grad * (tangent_x - x0) + y0
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'$f(x) = x^2$', color='#6C8EBF', linewidth=2)
    plt.plot(tangent_x, tangent_y, '--', color='#666666', label='Tangent (Slope > 0)')
    plt.scatter([x0], [y0], color='#B85450', s=100, zorder=5, label='Current Point')
    
    # Gradient arrow (Positive direction)
    plt.arrow(x0, y0, 0.5, 0.5 * grad, head_width=0.1, head_length=0.15, fc='#D6B656', ec='#D6B656', label='Gradient Direction')

    # Negative Gradient arrow (Descent direction)
    # We want to move along x-axis. Negative gradient is -grad.
    step_size = 0.8
    new_x = x0 - step_size * (grad / abs(grad)) # Normalized step for visualization
    
    plt.annotate('', xy=(new_x, y0), xytext=(x0, y0),
                 arrowprops=dict(facecolor='#82B366', shrink=0.05, width=2))
    plt.text((x0 + new_x)/2, y0 - 0.5, r'Negative Gradient $-\nabla f$', color='#82B366', ha='center', fontsize=12, fontweight='bold')

    plt.title('Why Negative Gradient? (Steepest Descent)', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_descent_1d.png'), dpi=300)
    plt.close()
    print("Generated gradient_descent_1d.png")

def plot_sgd_trajectory_3d():
    """
    Plot a 3D loss surface with a noisy SGD trajectory.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    X = np.arange(-2, 2, 0.1)
    Y = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 + Y**2  # Simple bowl

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.6, edgecolor='none')
    
    # Generate SGD path (noisy)
    np.random.seed(42)
    path_x = [1.8]
    path_y = [1.8]
    path_z = [path_x[0]**2 + path_y[0]**2]
    
    lr = 0.1
    for _ in range(20):
        curr_x = path_x[-1]
        curr_y = path_y[-1]
        
        # True gradient
        grad_x = 2 * curr_x
        grad_y = 2 * curr_y
        
        # Add noise to simulate stochastic gradient
        noise_x = np.random.normal(0, 1.5)
        noise_y = np.random.normal(0, 1.5)
        
        # SGD update
        new_x = curr_x - lr * (grad_x + noise_x)
        new_y = curr_y - lr * (grad_y + noise_y)
        
        path_x.append(new_x)
        path_y.append(new_y)
        path_z.append(new_x**2 + new_y**2)

    # Plot path
    ax.plot(path_x, path_y, path_z, color='#B85450', marker='o', markersize=4, linestyle='-', linewidth=2, label='SGD Path (Noisy)')
    ax.scatter([0], [0], [0], color='#82B366', s=100, marker='*', label='Global Minimum')

    ax.set_title('Stochastic Gradient Descent (SGD) Trajectory', fontsize=14)
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sgd_trajectory_3d.png'), dpi=300)
    plt.close()
    print("Generated sgd_trajectory_3d.png")

def plot_learning_rate_comparison():
    """
    Plot comparison of small, proper, and large learning rates.
    """
    x = np.linspace(-2.5, 2.5, 100)
    y = x**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Case 1: Small Learning Rate
    ax1 = axes[0]
    ax1.plot(x, y, color='#6C8EBF', alpha=0.5)
    
    start_x = -2.0
    lr_small = 0.05
    path_x = [start_x]
    path_y = [start_x**2]
    for _ in range(10):
        curr = path_x[-1]
        step = -lr_small * (2 * curr)
        nxt = curr + step
        path_x.append(nxt)
        path_y.append(nxt**2)
    
    ax1.plot(path_x, path_y, 'o-', color='#DAE8FC', markeredgecolor='#6C8EBF', label='Small LR (Slow)')
    ax1.set_title('Small Learning Rate (The Ant)', fontsize=14)
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Case 2: Large Learning Rate (Overshooting/Diverging)
    ax2 = axes[1]
    ax2.plot(x, y, color='#6C8EBF', alpha=0.5)
    
    start_x = -0.5 # Start closer to see the jump
    lr_large = 1.05 # Large enough to diverge for x^2 (since grad is 2x, update is x - 1.05*2x = -1.1x)
    path_x = [start_x]
    path_y = [start_x**2]
    for _ in range(4):
        curr = path_x[-1]
        step = -lr_large * (2 * curr)
        nxt = curr + step
        path_x.append(nxt)
        path_y.append(nxt**2)
        
    ax2.plot(path_x, path_y, 'o-', color='#F8CECC', markeredgecolor='#B85450', linewidth=2, label='Large LR (Unstable)')
    
    # Add arrows to show oscillation
    for i in range(len(path_x)-1):
        ax2.annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]),
                     arrowprops=dict(arrowstyle='->', color='#B85450'))

    ax2.set_title('Large Learning Rate (The Giant)', fontsize=14)
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'), dpi=300)
    plt.close()
    print("Generated learning_rate_comparison.png")

if __name__ == "__main__":
    plot_negative_gradient_1d()
    plot_sgd_trajectory_3d()
    plot_learning_rate_comparison()

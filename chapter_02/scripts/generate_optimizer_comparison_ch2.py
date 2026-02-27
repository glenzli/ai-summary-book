
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def loss_function(x, y):
    """
    Anisotropic quadratic loss: L = 0.5 * x^2 + 10 * y^2
    Represents a valley where gradient in y is much larger than x.
    """
    return 0.5 * x**2 + 10 * y**2

def gradient(x, y):
    return np.array([x, 20 * y])

def plot_trajectories(trajectories_dict, title, filename, show_contour=True):
    """
    Plots multiple optimization trajectories on a 2D contour.
    
    Args:
        trajectories_dict: Dictionary {Name: PathArray}
        title: Plot title
        filename: Output filename
    """
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Contour Background
    if show_contour:
        # Use a neutral Morandi-like grey/blue for contour
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), colors='#E1D5E7', alpha=0.5, linewidths=1)
        # Add a subtle shading
        ax.contourf(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='Greys', alpha=0.1)
    
    # Map input names to colors (fallback to black if not found)
    # Custom mapping for this specific script usage
    style_map = {
        'SGD': {'color': '#EA6B66', 'style': '--', 'marker': 'o', 'ms': 3},  # Red dashed
        'Momentum': {'color': '#6C8EBF', 'style': '-', 'marker': '^', 'ms': 4}, # Blue solid
        'RMSProp': {'color': '#9673A6', 'style': '-', 'marker': 's', 'ms': 4},  # Purple solid
        'Adam': {'color': '#82B366', 'style': '-', 'marker': '*', 'ms': 5}      # Green solid
    }

    for name, path in trajectories_dict.items():
        path = np.array(path)
        style = style_map.get(name, {'color': 'black', 'style': '-', 'marker': 'x', 'ms': 4})
        
        ax.plot(path[:, 0], path[:, 1], 
                linestyle=style['style'], 
                color=style['color'], 
                marker=style['marker'], 
                markersize=style['ms'], 
                linewidth=1.5, 
                label=name,
                alpha=0.9)
        
        # Mark start point clearly
        ax.plot(path[0, 0], path[0, 1], 'X', color='black', markersize=8, alpha=0.5)

    # Mark Global Min
    ax.plot(0, 0, 'P', color='#D6B656', markersize=12, label='Global Min', markeredgecolor='black')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Parameter x (Small Gradient)')
    ax.set_ylabel('Parameter y (Large Gradient)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # Set fixed limits for consistency across plots
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Generated {filename}")

def run_optimizers():
    # Initial point
    start_pos = np.array([-5.5, 2.5]) # Start further out
    iterations = 50
    
    # --- 1. SGD ---
    # Need a learning rate that shows oscillation but doesn't explode
    lr_sgd = 0.095 
    path_sgd = [start_pos]
    curr = start_pos.copy()
    for _ in range(iterations):
        grad = gradient(curr[0], curr[1])
        curr = curr - lr_sgd * grad
        path_sgd.append(curr.copy())
    
    # --- 2. Momentum ---
    # Standard Momentum: v = beta*v + grad; w = w - lr*v
    path_mom = [start_pos]
    curr = start_pos.copy()
    v = np.zeros_like(curr)
    beta = 0.9
    lr_mom = 0.01 # Smaller LR needed for standard momentum
    for _ in range(iterations):
        grad = gradient(curr[0], curr[1])
        v = beta * v + grad
        curr = curr - lr_mom * v
        path_mom.append(curr.copy())

    # --- 3. RMSProp ---
    path_rms = [start_pos]
    curr = start_pos.copy()
    s = np.zeros_like(curr)
    gamma = 0.99
    epsilon = 1e-8
    lr_rms = 0.5 # Larger LR allowed
    for _ in range(iterations):
        grad = gradient(curr[0], curr[1])
        s = gamma * s + (1 - gamma) * (grad**2)
        curr = curr - lr_rms * grad / (np.sqrt(s) + epsilon)
        path_rms.append(curr.copy())

    # --- 4. Adam ---
    path_adam = [start_pos]
    curr = start_pos.copy()
    m = np.zeros_like(curr)
    v = np.zeros_like(curr)
    beta1 = 0.9
    beta2 = 0.999
    lr_adam = 0.5
    for t in range(1, iterations + 1):
        grad = gradient(curr[0], curr[1])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        curr = curr - lr_adam * m_hat / (np.sqrt(v_hat) + epsilon)
        path_adam.append(curr.copy())

    # --- Generate Plots ---
    
    # 1. SGD vs Momentum
    plot_trajectories(
        {'SGD': path_sgd, 'Momentum': path_mom},
        'SGD vs Momentum (Inertia Smoothing)',
        'sgd_vs_momentum.png'
    )
    
    # 2. SGD vs RMSProp
    plot_trajectories(
        {'SGD': path_sgd, 'RMSProp': path_rms},
        'SGD vs RMSProp (Adaptive Scaling)',
        'sgd_vs_rmsprop.png'
    )
    
    # 3. SGD vs Adam (Optional, but user asked for "comparison all" and specific vs plots)
    # The user specifically mentioned moving sgd_vs_momentum, sgd_vs_rmsprop, and optimization_comparison_all
    
    # 4. All Combined (The "Family Portrait")
    plot_trajectories(
        {'SGD': path_sgd, 'Momentum': path_mom, 'RMSProp': path_rms, 'Adam': path_adam},
        'Optimization Algorithms Comparison',
        'optimizer_comparison_all.png'
    )

if __name__ == "__main__":
    run_optimizers()

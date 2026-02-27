import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_radius_impact():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    np.random.seed(42)

    # --- Subplot 1: Large Radius (Unnormalized) ---
    ax1 = axes[0]
    
    # Generate scattered data far from origin or spread out
    # Class 1: centered at (5, 5), spread 2
    # Class 2: centered at (8, 8), spread 2
    # This makes R large (distance from origin (0,0))
    
    N = 20
    X1 = np.random.randn(N, 2) + np.array([5, 5])
    X2 = np.random.randn(N, 2) + np.array([10, 10])
    
    # Calculate R (max distance from origin)
    all_X = np.vstack([X1, X2])
    R_val = np.max(np.linalg.norm(all_X, axis=1))
    
    # Plot data
    ax1.scatter(X1[:, 0], X1[:, 1], color='#6C8EBF', s=50, alpha=0.6, label='Class +1')
    ax1.scatter(X2[:, 0], X2[:, 1], color='#B85450', s=50, alpha=0.6, label='Class -1')
    
    # Draw enclosing circle for R
    circle = plt.Circle((0, 0), R_val, color='#D6B656', fill=False, linestyle='--', linewidth=1.5, label='Radius R')
    ax1.add_patch(circle)
    ax1.plot([0, all_X[0,0]], [0, all_X[0,1]], '--', color='#D6B656', alpha=0.5) # Radius line example
    ax1.text(all_X[0,0]/2, all_X[0,1]/2 + 1, f'R ≈ {R_val:.1f}', color='#D6B656', fontweight='bold')
    
    # Simulate "Oscillating" Updates (Schematic)
    # Start w at origin
    w_path = np.array([[0, 0], [2, 8], [8, 4], [4, 10], [10, 6]]) # Jerky path
    ax1.plot(w_path[:, 0], w_path[:, 1], 'o-', color='black', linewidth=2, label='Weight Updates')
    
    ax1.set_title(r'Large Radius $R$ (Unnormalized) $\rightarrow$ Unstable Updates', fontsize=14)
    ax1.set_xlim(-5, 18)
    ax1.set_ylim(-5, 18)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.3)

    # --- Subplot 2: Small Radius (Normalized) ---
    ax2 = axes[1]
    
    # Normalize data: Zero mean, divide by max norm to fit in unit circle roughly
    mean = np.mean(all_X, axis=0)
    centered_X = all_X - mean
    max_norm = np.max(np.linalg.norm(centered_X, axis=1))
    normalized_X = centered_X / max_norm # Now R = 1
    
    X1_norm = normalized_X[:N]
    X2_norm = normalized_X[N:]
    
    # Plot data
    ax2.scatter(X1_norm[:, 0], X1_norm[:, 1], color='#6C8EBF', s=50, alpha=0.6)
    ax2.scatter(X2_norm[:, 0], X2_norm[:, 1], color='#B85450', s=50, alpha=0.6)
    
    # Draw enclosing circle for R
    circle2 = plt.Circle((0, 0), 1.0, color='#82B366', fill=False, linestyle='--', linewidth=1.5, label='Radius R=1')
    ax2.add_patch(circle2)
    ax2.text(0.5, 0.5, 'R = 1', color='#82B366', fontweight='bold')
    
    # Simulate "Smooth" Updates (Schematic)
    # Start w at origin, move smoothly towards separation
    w_path_smooth = np.array([[0, 0], [0.2, 0.2], [0.4, 0.3], [0.5, 0.5]]) 
    ax2.plot(w_path_smooth[:, 0], w_path_smooth[:, 1], 'o-', color='black', linewidth=2, label='Weight Updates')
    
    ax2.set_title(r'Small Radius $R$ (Normalized) $\rightarrow$ Stable Convergence', fontsize=14)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_radius_normalization.png'), dpi=300)
    print("Generated data_radius_normalization.png")

if __name__ == "__main__":
    plot_radius_impact()

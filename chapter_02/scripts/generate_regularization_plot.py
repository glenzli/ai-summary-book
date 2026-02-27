
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def plot_regularization():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Grid for contour plot
    w1 = np.linspace(-3, 3, 500)
    w2 = np.linspace(-3, 3, 500)
    W1, W2 = np.meshgrid(w1, w2)
    
    # Hypothetical Loss Function (Ellipses centered at (1.5, 1.5))
    # Loss = (w1 - 1.5)^2 + 2 * (w2 - 1.5)^2
    Loss = (W1 - 1.5)**2 + 2 * (W2 - 1.5)**2
    
    # --- L1 Regularization (Lasso) ---
    ax1 = axes[0]
    # L1 Constraint: |w1| + |w2| <= 1
    l1_val = np.abs(W1) + np.abs(W2)
    
    # Plot Loss Contours
    levels = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0]
    ax1.contour(W1, W2, Loss, levels=levels, colors='#6C8EBF', alpha=0.8) # Blue
    
    # Plot L1 Constraint (Diamond)
    ax1.contourf(W1, W2, l1_val, levels=[0, 1], colors=['#FFF2CC'], alpha=0.5) # Yellow fill
    ax1.contour(W1, W2, l1_val, levels=[1], colors=['#D6B656'], linewidths=2) # Yellow border
    
    # Touch Point (Corner)
    ax1.scatter([1], [0], color='#B85450', s=100, zorder=5, label='Solution (Sparse)')
    
    ax1.set_title("L1 Regularization (Lasso)\nDiamond Constraint → Sparsity", fontsize=14)
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    # Increase limits to prevent cropping
    ax1.set_xlim(-2.2, 2.7)
    ax1.set_ylim(-2.2, 3.5)
    ax1.legend()

    # --- L2 Regularization (Ridge) ---
    ax2 = axes[1]
    # L2 Constraint: w1^2 + w2^2 <= 1
    l2_val = W1**2 + W2**2
    
    # Plot Loss Contours
    ax2.contour(W1, W2, Loss, levels=levels, colors='#6C8EBF', alpha=0.8)
    
    # Plot L2 Constraint (Circle)
    ax2.contourf(W1, W2, l2_val, levels=[0, 1], colors=['#DAE8FC'], alpha=0.5) # Blue fill
    ax2.contour(W1, W2, l2_val, levels=[1], colors=['#6C8EBF'], linewidths=2) # Blue border
    
    # Touch Point (Tangent) - Approximation for visual
    # Calculate tangent point roughly
    # Gradient of Circle is (2w1, 2w2). Gradient of Loss is (2(w1-1.5), 4(w2-1.5))
    # Visual approximation: usually somewhere along the line to center but pulled by ellipse shape
    tangent_w1, tangent_w2 = 0.8, 0.6 
    ax2.scatter([tangent_w1], [tangent_w2], color='#B85450', s=100, zorder=5, label='Solution (Non-sparse)')
    
    ax2.set_title("L2 Regularization (Ridge)\nCircle Constraint → Small Weights", fontsize=14)
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    # Increase limits to prevent cropping
    ax2.set_xlim(-2.2, 2.7)
    ax2.set_ylim(-2.2, 3.5)
    ax2.legend()
    
    plt.tight_layout(pad=3.0) # Add padding to layout
    plt.savefig('chapter_02/images/regularization_geometry.png', dpi=150, bbox_inches='tight', pad_inches=0.3)

if __name__ == "__main__":
    plot_regularization()

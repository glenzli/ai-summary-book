import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

# Ensure directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_vc_dimension():
    """
    Visualizes 3 points being shattered by a linear classifier,
    and 4 points (XOR) that cannot be shattered.
    """
    # utils.plot_style is automatically applied
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Left: Shattering 3 points ---
    ax1 = axes[0]
    
    # Define 3 points
    points_3 = np.array([[1, 1], [2, 3], [3, 1]])
    
    # Draw points
    ax1.scatter(points_3[:, 0], points_3[:, 1], s=150, c='#E1D5E7', edgecolors='#9673A6', linewidth=2, zorder=10)
    for i, p in enumerate(points_3):
        ax1.text(p[0]+0.15, p[1], f'$x_{{{i+1}}}$', fontsize=14)

    # Draw a few sample separating lines
    x_range = np.linspace(0, 4, 100)
    
    # Line 1: Separates x2 from (x1, x3)
    ax1.plot(x_range, np.ones_like(x_range)*2, '--', color='#6C8EBF', alpha=0.6, label='Line A')
    
    # Line 2: Separates x1 from (x2, x3)
    # y = 2x - 2 (approx)
    ax1.plot(x_range, 2*x_range - 2, '--', color='#B85450', alpha=0.6, label='Line B')
    
    ax1.set_title('VC=3: 3 Points can be fully shattered\n(Any labeling has a separating line)', fontsize=12)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.grid(True, alpha=0.3)
    
    # --- Right: XOR problem (4 points) ---
    ax2 = axes[1]
    
    # Define 4 points (XOR configuration)
    points_4 = np.array([[1, 1], [1, 3], [3, 1], [3, 3]])
    labels = [0, 1, 1, 0] # XOR labels
    colors = ['#B85450' if l==0 else '#6C8EBF' for l in labels] # Red for 0, Blue for 1
    
    # Draw points with colors
    ax2.scatter(points_4[:, 0], points_4[:, 1], s=150, c=colors, zorder=10, edgecolors='black')
    
    # Add labels text
    ax2.text(0.8, 0.8, '0 (Red)', color='#B85450', fontweight='bold')
    ax2.text(0.8, 3.2, '1 (Blue)', color='#6C8EBF', fontweight='bold')
    ax2.text(2.8, 0.8, '1 (Blue)', color='#6C8EBF', fontweight='bold')
    ax2.text(2.8, 3.2, '0 (Red)', color='#B85450', fontweight='bold')

    # Draw "Impossible" annotation
    ax2.text(2, 2, 'No linear line can\nseparate Red from Blue!', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='#FFF2CC', alpha=0.8, edgecolor='#D6B656'))
    
    ax2.set_title('VC < 4: 4 Points (XOR) cannot be shattered\n(Linear Classifier Fails)', fontsize=12)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vc_dimension.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_vc_dimension()
    print("VC Dimension image generated successfully.")

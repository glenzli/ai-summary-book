import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def plot_lora_matrix():
    # Setup styling
    # utils.plot_style is automatically applied
    
    fig = plt.figure(figsize=(12, 6))
    
    # 2D representation of Matrices
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Define box drawing helper
    def draw_matrix(x, y, w, h, label, color, text_label):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        # Center text
        ax.text(x + w/2, y + h/2, text_label, ha='center', va='center', fontsize=12, fontweight='bold', color='black')
        # Label below
        ax.text(x + w/2, y - 0.5, label, ha='center', va='top', fontsize=10)

    # Coordinates
    start_x = 0
    
    # 1. Pre-trained Weights W (Frozen)
    draw_matrix(start_x, 2, 4, 6, "Pre-trained Weights\n$W$ (d x d)\n[FROZEN]", '#E1D5E7', 'W')
    
    # Plus sign
    ax.text(start_x + 5, 5, "+", fontsize=20, ha='center')
    
    # 2. LoRA Adapter (A x B)
    # Matrix A (d x r)
    draw_matrix(start_x + 6, 2, 1, 6, "Matrix A\n(d x r)\n[Trainable]", '#DAE8FC', 'A')
    
    # Multiplication sign
    ax.text(start_x + 7.5, 5, "×", fontsize=20, ha='center')
    
    # Matrix B (r x d)
    # Note: Usually B is d x r and A is r x k, but typically presented as W + BA where B is d x r (up) and A is r x d (down)
    # Standard LoRA: h = Wx + BAx. B is d_out x r. A is r x d_in.
    # Let's visualize B * A
    
    # Let's adjust for visual clarity: Vertical tall matrix * Horizontal wide matrix
    # A: Tall (d x r) ? No, A is usually the down-projection (r x d) and B is up-projection (d x r).
    # Let's visualize standard convention: \Delta W = B @ A
    # B: d x r (Tall, Thin)
    # A: r x d (Short, Wide)
    
    # B (Tall)
    draw_matrix(start_x + 8, 2, 1.5, 6, "Matrix B\n(d x r)\n[Trainable]", '#DAE8FC', 'B')
    
    ax.text(start_x + 10, 5, "×", fontsize=20, ha='center')
    
    # A (Wide)
    draw_matrix(start_x + 11, 4, 4, 1.5, "Matrix A\n(r x d)\n[Trainable]", '#DAE8FC', 'A')
    
    # Equals
    ax.text(start_x + 16, 5, "=", fontsize=20, ha='center')
    
    # Result
    draw_matrix(start_x + 17, 2, 4, 6, "Effective Update\n$\Delta W$\n(d x d)", '#F8CECC', 'BA')

    # Add descriptive text
    ax.text(start_x + 10.5, 9, "LoRA: Low-Rank Adaptation", fontsize=16, ha='center', fontweight='bold')
    ax.text(start_x + 10.5, 8, "Instead of training W directly, we train low-rank matrices A and B.\nRank r << d (e.g., r=8, d=4096)", fontsize=12, ha='center')

    # Set limits
    ax.set_xlim(-1, 23)
    ax.set_ylim(0, 10)
    
    output_path = 'chapter_05/images/lora_diagram.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_lora_matrix()

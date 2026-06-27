import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def plot_scaling_law():
    # Define Compute (PF-days or FLOPs) range
    # Log scale range
    compute = np.logspace(18, 24, 100)  # 10^18 to 10^24 FLOPs
    
    # Power law: L(C) = (C_c / C)^alpha
    # Hypothetical parameters for illustration based on Kaplan et al. 2020
    # L(C) is proportional to C^-0.076 roughly (alpha)
    
    # Let's simulate a curve: Loss = A * C^(-alpha) + Irreducible Loss
    alpha = 0.076
    A = 10**2.5 # Arbitrary constant for scaling
    irreducible_loss = 1.5
    
    loss = A * (compute ** -alpha) + irreducible_loss
    
    # Create plot with Morandi style
    # utils.plot_style is automatically applied
    plt.figure(figsize=(10, 6))
    
    # Plot line
    plt.loglog(compute, loss, linewidth=3, color='#6C8EBF', label='Power Law Trend')
    
    # Annotations
    plt.title('Scaling Laws: Test Loss vs. Compute', fontsize=14, pad=20)
    plt.xlabel('Compute (FLOPs)', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    
    # Highlight regions
    # GPT-2 like
    c_gpt2 = 10**20
    l_gpt2 = A * (c_gpt2 ** -alpha) + irreducible_loss
    plt.scatter([c_gpt2], [l_gpt2], color='#B85450', s=100, zorder=5)
    plt.text(c_gpt2 * 1.5, l_gpt2, 'GPT-2 Scale', fontsize=10, verticalalignment='center')
    
    # GPT-3 like
    c_gpt3 = 10**23
    l_gpt3 = A * (c_gpt3 ** -alpha) + irreducible_loss
    plt.scatter([c_gpt3], [l_gpt3], color='#82B366', s=100, zorder=5)
    plt.text(c_gpt3 * 1.5, l_gpt3, 'GPT-3 Scale', fontsize=10, verticalalignment='center')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save
    output_path = 'chapter_04/images/scaling_law_plot.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_scaling_law()

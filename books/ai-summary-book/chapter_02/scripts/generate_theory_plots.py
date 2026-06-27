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

def plot_bias_variance():
    x = np.linspace(0, 10, 100)
    
    # Model Complexity vs Error
    bias_sq = 10 * np.exp(-0.5 * x) + 0.5  # Decreases with complexity
    variance = 0.5 * np.exp(0.4 * x)       # Increases with complexity
    noise = np.ones_like(x) * 1.5          # Constant noise
    total_error = bias_sq + variance + noise
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, bias_sq, label='Bias²', color='#6C8EBF', linewidth=2.5, linestyle='--')
    plt.plot(x, variance, label='Variance', color='#B85450', linewidth=2.5, linestyle='--')
    plt.plot(x, total_error, label='Total Error', color='#000000', linewidth=3)
    plt.plot(x, noise, label='Irreducible Error', color='#666666', linewidth=1.5, linestyle=':')
    
    # Find sweet spot
    min_idx = np.argmin(total_error)
    plt.axvline(x[min_idx], color='#82B366', linestyle='--', alpha=0.5)
    plt.text(x[min_idx], plt.ylim()[1]*0.95, 'Sweet Spot\n(Optimal)', 
             ha='center', va='top', color='#82B366', fontweight='bold')

    plt.title('Bias-Variance Tradeoff', fontsize=14, pad=20)
    plt.xlabel('Model Complexity', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(frameon=True, facecolor='#F5F5F5')
    plt.grid(True, alpha=0.3)
    
    # Annotate regions
    plt.text(1, total_error.max()*0.8, 'Underfitting\n(High Bias)', ha='center', color='#6C8EBF')
    plt.text(9, total_error.max()*0.8, 'Overfitting\n(High Variance)', ha='center', color='#B85450')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bias_variance_tradeoff.png", dpi=300)
    plt.close()

def plot_universal_approximation():
    """
    Shows how ReLUs combine to form a bump.
    """
    x = np.linspace(-2, 4, 400)
    
    def relu(x):
        return np.maximum(0, x)
    
    # Construct a bump: ReLU(x) - ReLU(x-1) - ReLU(x-2) + ReLU(x-3)
    # This is a trapezoidal bump
    y1 = relu(x)
    y2 = -relu(x - 1)
    y3 = -relu(x - 2)
    y4 = relu(x - 3)
    
    y_total = y1 + y2 + y3 + y4
    
    plt.figure(figsize=(10, 6))
    
    # Plot components
    plt.plot(x, y1, '--', alpha=0.3, label='ReLU(x)')
    plt.plot(x, y2, '--', alpha=0.3, label='-ReLU(x-1)')
    plt.plot(x, y3, '--', alpha=0.3, label='-ReLU(x-2)')
    plt.plot(x, y4, '--', alpha=0.3, label='ReLU(x-3)')
    
    # Plot Result
    plt.plot(x, y_total, color='#B85450', linewidth=3, label='Combined Bump Function')
    
    plt.fill_between(x, y_total, color='#F8CECC', alpha=0.3)
    
    plt.title('Universal Approximation Construction\n(Combining ReLUs to form a local "Bump")', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/universal_approximation_bump.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating bias_variance_tradeoff.png...")
    plot_bias_variance()
    print("Generating universal_approximation_bump.png...")
    plot_universal_approximation()
    print("All images generated successfully.")

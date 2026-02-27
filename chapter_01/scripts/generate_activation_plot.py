import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_activation(x, y, name, formula, filename, ylim=None):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2)
    plt.title(name, fontsize=14)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    if ylim:
        plt.ylim(ylim)
        
    # Add formula text
    plt.text(0.05, 0.95, formula, transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Image saved to {output_path}")
    plt.close()

x = np.linspace(-5, 5, 200)

# 1. Sigmoid
y_sigmoid = 1 / (1 + np.exp(-x))
plot_activation(x, y_sigmoid, 'Sigmoid', r'$\sigma(x) = \frac{1}{1+e^{-x}}$', 'sigmoid.png', ylim=(-0.1, 1.1))

# 2. Tanh
y_tanh = np.tanh(x)
plot_activation(x, y_tanh, 'Tanh', r'$\sigma(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$', 'tanh.png', ylim=(-1.1, 1.1))

# 3. ReLU
y_relu = np.maximum(0, x)
plot_activation(x, y_relu, 'ReLU', r'$\sigma(x) = \max(0, x)$', 'relu.png', ylim=(-1.0, 5.0))

# 4. Leaky ReLU
y_leaky = np.maximum(0.1 * x, x)
plot_activation(x, y_leaky, 'Leaky ReLU', r'$\sigma(x) = \max(0.1x, x)$', 'leaky_relu.png', ylim=(-1.0, 5.0))

# 5. GELU
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

y_gelu = gelu(x)
plot_activation(x, y_gelu, 'GELU', r'$GELU(x) \approx 0.5x(1+\tanh(\ldots))$', 'gelu.png', ylim=(-1.0, 5.0))

# 6. Swish
def swish(x, beta=1):
    return x * (1 / (1 + np.exp(-beta * x)))

y_swish = swish(x)
plot_activation(x, y_swish, 'Swish', r'$Swish(x) = x \cdot \sigma(x)$', 'swish.png', ylim=(-1.0, 5.0))

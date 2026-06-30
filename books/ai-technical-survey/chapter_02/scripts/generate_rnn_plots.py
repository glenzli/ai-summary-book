
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_vanishing_gradient():
    """
    Visualizes the exponential decay/explosion of gradients over time steps.
    y = lambda^t
    """
    t = np.arange(0, 100, 1)
    lambdas = [0.9, 1.0, 1.1]
    colors = ['#B85450', '#82B366', '#6C8EBF'] # Red, Green, Blue
    labels = [r'$\lambda = 0.9$ (Vanishing)', r'$\lambda = 1.0$ (Stable)', r'$\lambda = 1.1$ (Exploding)']

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for lam, col, lab in zip(lambdas, colors, labels):
        y = np.power(lam, t)
        ax.plot(t, y, label=lab, color=col, linewidth=2)

    ax.set_title('Gradient Signal over Time Steps (Geometric Growth/Decay)', fontsize=12)
    ax.set_xlabel('Time Steps (t)')
    ax.set_ylabel(r'Gradient Magnitude ($\lambda^t$)')
    ax.set_yscale('log') # Use log scale to show the dramatic difference
    ax.set_ylim(1e-5, 1e5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vanishing_gradient.png'), dpi=200)
    print("Generated vanishing_gradient.png")

def plot_activation_gradients():
    """
    Visualizes the derivatives of Sigmoid, Tanh, and ReLU.
    Shows that Sigmoid/Tanh derivatives are always < 1 (or <= 0.25 for Sigmoid).
    """
    x = np.linspace(-5, 5, 200)
    
    # Functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    
    # Derivatives
    d_sigmoid = sigmoid * (1 - sigmoid)
    d_tanh = 1 - tanh**2
    d_relu = np.where(x > 0, 1.0, 0.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(x, d_sigmoid, label=r"Sigmoid' (max 0.25)", color='#B85450', linestyle='--', linewidth=2)
    ax.plot(x, d_tanh, label=r"Tanh' (max 1.0)", color='#D6B656', linestyle='-', linewidth=2)
    ax.plot(x, d_relu, label=r"ReLU' (max 1.0, constant)", color='#82B366', linestyle='-', linewidth=2)
    
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Gradient = 1.0')
    
    ax.set_title('Derivatives of Activation Functions\n(Source of Vanishing Gradient)', fontsize=12)
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('Derivative f\'(x)')
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_gradients.png'), dpi=200)
    print("Generated activation_gradients.png")

if __name__ == "__main__":
    plot_vanishing_gradient()
    plot_activation_gradients()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def simulate_network(init_mode, activation_fn, num_layers=10, hidden_dim=500):
    # Input data: standard normal
    dims = [hidden_dim] * (num_layers + 1)
    x = np.random.randn(1000, hidden_dim)
    
    activations = {}
    
    for i in range(num_layers):
        din = dims[i]
        dout = dims[i+1]
        
        # Initialization logic
        if init_mode == 'small':
            W = np.random.randn(din, dout) * 0.01
        elif init_mode == 'large':
            W = np.random.randn(din, dout) * 1.0
        elif init_mode == 'xavier':
            # Xavier: var = 1/n (simplified for fan_in)
            W = np.random.randn(din, dout) / np.sqrt(din)
        elif init_mode == 'he':
            # He: var = 2/n
            W = np.random.randn(din, dout) * np.sqrt(2.0/din)
        else:
            raise ValueError("Unknown init mode")
            
        # Forward pass
        z = np.dot(x, W)
        x = activation_fn(z)
        
        activations[i] = x
        
    return activations

def plot_weight_initialization():
    # utils.plot_style is automatically applied
    
    # Configuration
    configs = [
        ('small', tanh, 'Small Random (0.01)\n(Vanishing Signal)', 'Tanh'),
        ('large', tanh, 'Large Random (1.0)\n(Saturation/Exploding)', 'Tanh'),
        ('xavier', tanh, 'Xavier Initialization\n(Stable)', 'Tanh'),
        ('he', relu, 'He Initialization\n(Stable for ReLU)', 'ReLU')
    ]
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharey='row')
    
    # Layers to visualize
    layer_indices = [0, 2, 4, 6, 9] 
    
    for row_idx, (init_mode, act_fn, title, act_name) in enumerate(configs):
        activations = simulate_network(init_mode, act_fn)
        
        for col_idx, layer_idx in enumerate(layer_indices):
            ax = axes[row_idx, col_idx]
            data = activations[layer_idx].flatten()
            
            # Plot histogram
            ax.hist(data, bins=30, range=(-1.5, 1.5) if act_name=='Tanh' else (-0.1, 2.5), 
                    color='#6C8EBF' if act_name=='Tanh' else '#82B366', 
                    edgecolor='none', alpha=0.8)
            
            # Styling
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Layer {layer_idx+1}", fontsize=12, fontweight='bold')
                
            # Add mean/std stats
            mean = np.mean(data)
            std = np.std(data)
            ax.text(0.95, 0.85, f"$\mu$={mean:.2f}\n$\sigma$={std:.2f}", 
                    transform=ax.transAxes, ha='right', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Row Label
        fig.text(0.08, 0.81 - row_idx * 0.21, title, ha='center', va='center', fontsize=12, fontweight='bold', rotation=0)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.05, hspace=0.4, wspace=0.1)
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '../images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'weight_initialization.png'), bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'weight_initialization.png')}")

if __name__ == "__main__":
    plot_weight_initialization()
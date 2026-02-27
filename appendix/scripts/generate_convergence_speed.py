import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def perceptron_train(X, y, max_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    bias = 0
    
    history = [] # Store (w, b) at each update
    history.append((w.copy(), bias))
    
    steps = 0
    converged = False
    
    for t in range(max_iter):
        mistake_found = False
        for i in range(n_samples):
            if y[i] * (np.dot(X[i], w) + bias) <= 0:
                w += y[i] * X[i]
                bias += y[i]
                history.append((w.copy(), bias))
                mistake_found = True
                steps += 1
                break # Simple PLA updates one at a time
        
        if not mistake_found:
            converged = True
            break
            
    return w, bias, history, steps

def plot_convergence_speed():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Common random seed
    np.random.seed(42)

    # --- Dataset 1: Large Margin ---
    # Class 1: [1, 1] to [2, 2]
    # Class 2: [4, 4] to [5, 5]
    # Separator around x+y=6 or similar. Distance is large.
    X1 = np.random.rand(5, 2) + np.array([0, 0])
    X2 = np.random.rand(5, 2) + np.array([4, 4])
    X_large = np.vstack([X1, X2])
    y_large = np.array([1]*5 + [-1]*5)
    
    w_large, b_large, hist_large, steps_large = perceptron_train(X_large, y_large)
    
    ax1 = axes[0]
    ax1.scatter(X1[:, 0], X1[:, 1], color='#6C8EBF', s=80, label='Class +1', edgecolor='k')
    ax1.scatter(X2[:, 0], X2[:, 1], color='#B85450', s=80, label='Class -1', edgecolor='k')
    
    # Plot history lines (faded)
    x_range = np.linspace(-1, 6, 100)
    
    # Plot a few intermediate lines
    colors = plt.cm.Greys(np.linspace(0.3, 0.8, len(hist_large)))
    for idx, (w, b) in enumerate(hist_large[1:]): # Skip initial zero weight
        if w[1] != 0:
            y_vals = -(w[0] * x_range + b) / w[1]
            # Clip y for better visuals
            ax1.plot(x_range, y_vals, color=colors[idx], alpha=0.5, linewidth=1)

    # Plot final line
    if w_large[1] != 0:
        y_final = -(w_large[0] * x_range + b_large) / w_large[1]
        ax1.plot(x_range, y_final, color='#82B366', linewidth=3, label='Final Separator')

    ax1.set_title(f'Large Margin: Converged in {steps_large} Steps', fontsize=14)
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 6)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.3)
    
    # --- Dataset 2: Small Margin ---
    # Shift Class 2 very close to Class 1
    # Class 1 ends around [1,1]
    # Class 2 starts around [1.2, 1.2]
    X1_small = np.random.rand(5, 2) + np.array([0, 0])
    X2_small = np.random.rand(5, 2) + np.array([1.2, 1.2]) # Much closer
    # Ensure linear separability manually if needed, but rand usually works here with small gap
    # Let's force a gap
    X2_small = X2_small + 0.2 
    
    X_small = np.vstack([X1_small, X2_small])
    y_small = np.array([1]*5 + [-1]*5)
    
    w_small, b_small, hist_small, steps_small = perceptron_train(X_small, y_small)
    
    ax2 = axes[1]
    ax2.scatter(X1_small[:, 0], X1_small[:, 1], color='#6C8EBF', s=80, edgecolor='k')
    ax2.scatter(X2_small[:, 0], X2_small[:, 1], color='#B85450', s=80, edgecolor='k')
    
    # Plot history lines (faded)
    # Since there are many steps, we subsample or just plot them all very thin
    colors = plt.cm.Greys(np.linspace(0.2, 0.6, len(hist_small)))
    for idx, (w, b) in enumerate(hist_small[1:]):
        if w[1] != 0:
            y_vals = -(w[0] * x_range + b) / w[1]
            ax2.plot(x_range, y_vals, color='gray', alpha=0.2, linewidth=0.5)

    # Plot final line
    if w_small[1] != 0:
        y_final = -(w_small[0] * x_range + b_small) / w_small[1]
        ax2.plot(x_range, y_final, color='#B85450', linewidth=3, label='Final Separator')

    ax2.set_title(f'Small Margin: Converged in {steps_small} Steps', fontsize=14)
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 4)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # Add text annotation about the "Struggle"
    ax2.text(2.5, 0.5, "Many adjustments\nneeded to fit\nin the narrow gap", 
             fontsize=12, color='#B85450', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_speed_gamma.png'), dpi=300)
    print("Generated convergence_speed_gamma.png")

if __name__ == "__main__":
    plot_convergence_speed()

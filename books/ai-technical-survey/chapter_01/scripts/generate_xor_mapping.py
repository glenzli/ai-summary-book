import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return np.maximum(0, x)

def plot_xor_mapping():
    # Input points
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0]) # XOR labels

    # Define weights and biases for hidden layer (as per text)
    # h1 = ReLU(x1 + x2)
    # h2 = ReLU(x1 + x2 - 1)
    
    H = np.zeros_like(X)
    for i in range(len(X)):
        x1, x2 = X[i]
        h1 = relu(x1 + x2)
        h2 = relu(x1 + x2 - 1)
        H[i] = [h1, h2]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#FFFFFF')

    # Styles
    colors = ['#B85450' if label == 0 else '#6C8EBF' for label in y] # Red for 0, Blue for 1
    markers = ['o', 's', 's', 'o'] # Circle for 0, Square for 1 (just distinct shapes)
    
    # 1. Input Space
    ax1.set_title("Original Input Space $(x_1, x_2)$", fontsize=14)
    ax1.set_xlabel("$x_1$", fontsize=12)
    ax1.set_ylabel("$x_2$", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    
    # Draw points
    for i in range(len(X)):
        ax1.scatter(X[i, 0], X[i, 1], c=colors[i], s=150, zorder=3, edgecolors='k')
        ax1.text(X[i, 0]+0.1, X[i, 1], f"({X[i, 0]}, {X[i, 1]})", fontsize=10)

    # 2. Hidden Space
    ax2.set_title("Hidden Feature Space $(h_1, h_2)$", fontsize=14)
    ax2.set_xlabel("$h_1 = \text{ReLU}(x_1+x_2)$", fontsize=12)
    ax2.set_ylabel("$h_2 = \text{ReLU}(x_1+x_2-1)$", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 1.5)

    # Draw separation line in Hidden Space: y = h1 - 2h2 = 0.5 -> h2 = 0.5*h1 - 0.25
    # Let's visualize the decision boundary h1 - 2h2 = 0.5
    h1_vals = np.linspace(-0.5, 2.5, 100)
    h2_vals = 0.5 * h1_vals - 0.25
    ax2.plot(h1_vals, h2_vals, '--', color='#82B366', linewidth=2, label='Decision Boundary\n$h_1 - 2h_2 = 0.5$')

    # Draw points
    # Note: [0,1] and [1,0] map to the same point [1,0]
    for i in range(len(H)):
        # Add slight jitter to separate overlapping points visually if needed, 
        # but here we want to show they overlap or are distinct.
        # Actually [0,1] and [1,0] map EXACTLY to [1,0].
        # We can draw them slightly offset or just one on top of another.
        
        offset = 0
        if i == 2: offset = 0.05 # Offset the second point slightly to show it exists
        
        ax2.scatter(H[i, 0], H[i, 1] + offset, c=colors[i], s=150, zorder=3, edgecolors='k')
        
        label_text = f"({int(H[i,0])}, {int(H[i,1])})"
        if i == 1: label_text += " & "
        if i == 2: continue # Handled by i=1 text roughly
        
        ax2.text(H[i, 0]+0.1, H[i, 1] + offset, label_text, fontsize=10)
        
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('chapter_01/images/xor_mapping_process.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_xor_mapping()

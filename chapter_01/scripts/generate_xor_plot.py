import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_logic_gates():
    """
    Plot AND, OR, XOR gates to show linear separability issues.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Logic outputs
    y_and = np.array([0, 0, 0, 1])
    y_or  = np.array([0, 1, 1, 1])
    y_xor = np.array([0, 1, 1, 0])
    
    titles = ['AND Gate (Linearly Separable)', 'OR Gate (Linearly Separable)', 'XOR Gate (Not Separable)']
    ys = [y_and, y_or, y_xor]
    
    # Decision boundaries (manual for illustration)
    # AND: x + y - 1.5 = 0
    # OR:  x + y - 0.5 = 0
    # XOR: Impossible
    
    x_line = np.linspace(-0.5, 1.5, 100)
    
    for i, ax in enumerate(axes):
        y_curr = ys[i]
        
        # Plot points
        # Class 0: Red, Class 1: Blue
        ax.scatter(X[y_curr==0][:, 0], X[y_curr==0][:, 1], color='#B85450', s=200, label='0', zorder=5, edgecolor='k')
        ax.scatter(X[y_curr==1][:, 0], X[y_curr==1][:, 1], color='#6C8EBF', s=200, label='1', zorder=5, edgecolor='k')
        
        # Grid and Limits
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        
        if i == 0: # AND
            # x + y = 1.5 => y = 1.5 - x
            ax.plot(x_line, 1.5 - x_line, color='#82B366', linewidth=3, linestyle='--')
            ax.fill_between(x_line, 1.5 - x_line, 2, color='#6C8EBF', alpha=0.1)
            ax.fill_between(x_line, -1, 1.5 - x_line, color='#B85450', alpha=0.1)
            
        elif i == 1: # OR
            # x + y = 0.5 => y = 0.5 - x
            ax.plot(x_line, 0.5 - x_line, color='#82B366', linewidth=3, linestyle='--')
            ax.fill_between(x_line, 0.5 - x_line, 2, color='#6C8EBF', alpha=0.1)
            ax.fill_between(x_line, -1, 0.5 - x_line, color='#B85450', alpha=0.1)
            
        elif i == 2: # XOR
            # No single line works.
            # Visualizing the impossibility
            ax.text(0.5, 0.5, '?', fontsize=50, color='gray', ha='center', va='center', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xor_problem.png'), dpi=300)
    print("Generated xor_problem.png")

if __name__ == "__main__":
    plot_logic_gates()

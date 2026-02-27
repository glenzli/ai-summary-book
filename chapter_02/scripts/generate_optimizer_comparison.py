import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def objective_function(x, y):
    # A function with a ravine/valley and potentially saddle point characteristics
    # Simple Quadratic Valley with high condition number
    return 0.1 * x**2 + 2 * y**2

def gradient(x, y):
    dx = 0.2 * x
    dy = 4 * y
    return np.array([dx, dy])

class Optimizer:
    def __init__(self, start_pos, lr=0.1):
        self.path = [start_pos]
        self.pos = np.array(start_pos, dtype=np.float64)
        self.lr = lr
        
    def step(self):
        pass

class SGD(Optimizer):
    def step(self):
        grad = gradient(self.pos[0], self.pos[1])
        self.pos -= self.lr * grad
        self.path.append(self.pos.copy())

class Momentum(Optimizer):
    def __init__(self, start_pos, lr=0.1, beta=0.9):
        super().__init__(start_pos, lr)
        self.beta = beta
        self.velocity = np.zeros(2)
        
    def step(self):
        grad = gradient(self.pos[0], self.pos[1])
        self.velocity = self.beta * self.velocity + self.lr * grad
        self.pos -= self.velocity
        self.path.append(self.pos.copy())

class RMSProp(Optimizer):
    def __init__(self, start_pos, lr=0.1, beta=0.9, epsilon=1e-8):
        super().__init__(start_pos, lr)
        self.beta = beta
        self.epsilon = epsilon
        self.s = np.zeros(2)
        
    def step(self):
        grad = gradient(self.pos[0], self.pos[1])
        self.s = self.beta * self.s + (1 - self.beta) * (grad ** 2)
        self.pos -= self.lr * grad / (np.sqrt(self.s) + self.epsilon)
        self.path.append(self.pos.copy())

class Adam(Optimizer):
    def __init__(self, start_pos, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(start_pos, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        
    def step(self):
        self.t += 1
        grad = gradient(self.pos[0], self.pos[1])
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        self.pos -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.path.append(self.pos.copy())

def plot_optimizer_comparison():
    # utils.plot_style is automatically applied
    
    # Setup Data
    start_pos = [-8.0, 2.0]
    iterations = 50
    
    # Run Optimizers
    # SGD needs small LR to not diverge in steep direction, but then is slow in flat direction
    sgd = SGD(start_pos, lr=0.4) 
    # Momentum helps
    momentum = Momentum(start_pos, lr=0.1, beta=0.9)
    # RMSProp adapts
    rmsprop = RMSProp(start_pos, lr=0.8) # RMSProp also allows larger LR
    # Adam adapts
    adam = Adam(start_pos, lr=0.8) # Adam usually needs larger LR in these toy examples to show speed visually
    
    for _ in range(iterations):
        sgd.step()
        momentum.step()
        rmsprop.step()
        adam.step()

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '../images')
    os.makedirs(output_dir, exist_ok=True)
        
    # --- 1. Generate 3D Landscape Plot ---
    # Coarse grid for 3D is fine
    x_3d = np.linspace(-10, 10, 100)
    y_3d = np.linspace(-5, 5, 100)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    Z_3d = objective_function(X_3d, Y_3d)
    
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot_surface(X_3d, Y_3d, Z_3d, cmap='Blues', alpha=0.6, rstride=5, cstride=5, edgecolor='none')
    ax_3d.view_init(elev=45, azim=-60)
    ax_3d.set_title("Optimization Landscape (3D)", pad=20)
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('Loss')
    
    def plot_path_3d(ax, optimizer, color, label):
        path = np.array(optimizer.path)
        z_path = objective_function(path[:, 0], path[:, 1])
        ax.plot(path[:, 0], path[:, 1], z_path + 2, color=color, linewidth=2, label=label, marker='.', markersize=4)
        ax.scatter(path[0,0], path[0,1], z_path[0]+2, color='black', s=50)
        ax.scatter(path[-1,0], path[-1,1], z_path[-1]+2, color=color, s=50, marker='*')

    plot_path_3d(ax_3d, sgd, '#B85450', 'SGD')       # Red
    plot_path_3d(ax_3d, momentum, '#D6B656', 'Momentum') # Yellow
    plot_path_3d(ax_3d, rmsprop, '#9673A6', 'RMSProp') # Purple
    plot_path_3d(ax_3d, adam, '#82B366', 'Adam')     # Green
    
    ax_3d.legend()
    plt.savefig(os.path.join(output_dir, 'optimizer_landscape_3d.png'), bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'optimizer_landscape_3d.png')}")
    plt.close(fig_3d)

    # --- 2. Generate 2D Trajectory Plot ---
    # Crop the view to focus on the trajectory details (x <= 3, |y| <= 3)
    # Start point is -8, so we keep the left bound at -10
    x_2d = np.linspace(-10, 3, 400) 
    y_2d = np.linspace(-3, 3, 400)
    X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
    Z_2d = objective_function(X_2d, Y_2d)

    fig_2d, ax_2d = plt.subplots(figsize=(10, 6)) # Adjusted aspect ratio for the cropped view
    # Use logarithmic levels to better visualize convergence near minimum
    levels = np.logspace(np.log10(0.01), np.log10(np.max(Z_2d)), 30)
    cp = ax_2d.contour(X_2d, Y_2d, Z_2d, levels=levels, cmap='Blues', alpha=0.6)
    
    def plot_path_2d(ax, optimizer, color, label):
        path = np.array(optimizer.path)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, label=label, marker='.', markersize=4)
        ax.scatter(path[0,0], path[0,1], color='black', s=50, zorder=5)
        ax.scatter(path[-1,0], path[-1,1], color=color, s=100, marker='*', zorder=5)

    plot_path_2d(ax_2d, sgd, '#B85450', 'SGD (Oscillation)')
    plot_path_2d(ax_2d, momentum, '#D6B656', 'Momentum (Inertia)')
    plot_path_2d(ax_2d, rmsprop, '#9673A6', 'RMSProp (Adaptive)')
    plot_path_2d(ax_2d, adam, '#82B366', 'Adam (Adaptive+Momentum)')
    
    ax_2d.set_title("Trajectory Comparison (2D Contour)")
    ax_2d.set_xlabel('x (Flat Direction)')
    ax_2d.set_ylabel('y (Steep Direction)')
    ax_2d.legend()
    ax_2d.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'optimizer_trajectory_2d.png'), bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'optimizer_trajectory_2d.png')}")
    plt.close(fig_2d)

if __name__ == "__main__":
    plot_optimizer_comparison()
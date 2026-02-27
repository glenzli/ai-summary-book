
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_path(ax, path, color, label, linestyle='-', marker_size=2):
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=color, label=label, linewidth=1.5, linestyle=linestyle, marker='.', markersize=marker_size)
    ax.plot(path[0, 0], path[0, 1], 'x', color='black', markersize=6, alpha=0.6) # Start

def run_scale_problem_momentum_only():
    """
    Scenario 1: Scale Imbalance (The Valley) - Badcase for Momentum
    Demonstrate that Momentum with small LR crawls in flat direction.
    """
    def loss_func(x, y): return 0.05 * x**2 + 5.0 * y**2 # Highly anisotropic
    def grad_func(x, y): return np.array([0.1 * x, 10.0 * y])
    
    start_pos = np.array([-4.5, 1.0])
    iterations = 50
    
    # Momentum with small LR
    path_mom = [start_pos]
    curr = start_pos.copy()
    v = np.zeros_like(curr)
    lr = 0.02 # Must be small to avoid explosion in Y
    beta = 0.9
    
    for _ in range(iterations):
        g = grad_func(curr[0], curr[1])
        v = beta * v + g
        curr = curr - lr * v
        path_mom.append(curr.copy())
        
    # Plotting
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # Use logarithmic levels for better visualization of the valley
    ax.contour(X, Y, Z, levels=np.logspace(-2, 2, 20), colors='#E1D5E7', alpha=0.8)
    ax.contourf(X, Y, Z, levels=np.logspace(-2, 2, 20), cmap='Greys', alpha=0.1)
    
    # Plot Momentum badcase
    plot_path(ax, path_mom, '#6C8EBF', 'Momentum (Forced Small LR)', marker_size=8)
    
    ax.set_title('Momentum Badcase: Scale Imbalance\n(Crawls in flat x-direction due to global small LR)', fontsize=11)
    ax.set_xlabel('Flat Direction (x)')
    ax.set_ylabel('Steep Direction (y)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'momentum_badcase_scale.png'), dpi=200)
    print("Generated momentum_badcase_scale.png")

def run_noise_problem_rmsprop_only():
    """
    Scenario 2: Noisy Gradients - Badcase for RMSProp
    Demonstrate that RMSProp jitters significantly without momentum.
    Comparing against SGD to show it's not just a scale issue, but a noise issue.
    Actually, just showing RMSProp jitter is enough as per user request.
    """
    np.random.seed(42)
    def loss_func(x, y): return x**2 + y**2
    def grad_func(x, y): 
        # Add random Gaussian noise to gradient
        noise = np.random.normal(0, 2.0, size=2)
        return np.array([2*x, 2*y]) + noise
    
    start_pos = np.array([-3.0, -3.0])
    iterations = 40
    
    # RMSProp (No Momentum)
    path_rms = [start_pos]
    curr = start_pos.copy()
    s = np.zeros_like(curr)
    lr = 0.1
    gamma = 0.9
    
    for _ in range(iterations):
        g = grad_func(curr[0], curr[1])
        s = gamma * s + (1 - gamma) * g**2
        curr = curr - lr * g / (np.sqrt(s) + 1e-8)
        path_rms.append(curr.copy())
        
    # Plotting
    x = np.linspace(-4, 2, 100)
    y = np.linspace(-4, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(X, Y, Z, levels=15, colors='#E1D5E7', alpha=0.8)
    
    plot_path(ax, path_rms, '#9673A6', 'RMSProp (No Inertia)', linestyle='--', marker_size=6)
    
    ax.set_title('RMSProp Badcase: Noisy Gradients\n(Jitters violently near optimum due to lack of momentum)', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmsprop_badcase_noise.png'), dpi=200)
    print("Generated rmsprop_badcase_noise.png")

def run_adam_vs_badcases():
    """
    Adam Grand Synthesis
    Comparing Adam against the badcases. 
    1. vs Scale Problem (Adam handles x-axis speed)
    2. vs Noise Problem (Adam handles smoothness)
    User asked for: Adam comparison against Momentum Badcase and Adaptive Badcase.
    Let's make two subplots or one combined plot if possible, but two is clearer.
    Actually, let's just create one plot showing Adam on the Scale problem and one on Noise problem.
    """
    
    # --- Plot 1: Adam on Scale Problem ---
    def loss_func_scale(x, y): return 0.05 * x**2 + 5.0 * y**2
    def grad_func_scale(x, y): return np.array([0.1 * x, 10.0 * y])
    
    start_pos = np.array([-4.5, 1.0])
    iterations = 50
    
    # Adam
    path_adam_scale = [start_pos]
    curr = start_pos.copy()
    m = np.zeros_like(curr); v = np.zeros_like(curr)
    lr = 0.1; beta1 = 0.9; beta2 = 0.999
    
    for t in range(1, iterations + 1):
        g = grad_func_scale(curr[0], curr[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        curr = curr - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        path_adam_scale.append(curr.copy())

    # Generate Momentum path again for comparison in this plot? 
    # User said: "Adam perfect solved... introduce Momentum badcase... comparison"
    # So yes, show Momentum failing vs Adam succeeding.
    path_mom_scale = [start_pos]
    curr = start_pos.copy()
    v_mom = np.zeros_like(curr)
    lr_mom = 0.02 # Constrained small LR
    for _ in range(iterations):
        g = grad_func_scale(curr[0], curr[1])
        v_mom = 0.9 * v_mom + g
        curr = curr - lr_mom * v_mom
        path_mom_scale.append(curr.copy())

    x = np.linspace(-5, 5, 100); y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y); Z = loss_func_scale(X, Y)
    
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.contour(X, Y, Z, levels=np.logspace(-2, 2, 20), colors='#E1D5E7', alpha=0.8)
    ax1.contourf(X, Y, Z, levels=np.logspace(-2, 2, 20), cmap='Greys', alpha=0.1)
    
    plot_path(ax1, path_mom_scale, '#6C8EBF', 'Momentum (Fails: Crawls)', marker_size=6)
    plot_path(ax1, path_adam_scale, '#82B366', 'Adam (Success: Fast)', marker_size=6)
    
    ax1.set_title('Adam vs Scale Imbalance\n(Adam adapts step size for fast convergence)', fontsize=11)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adam_vs_scale.png'), dpi=200)
    
    # --- Plot 2: Adam on Noise Problem ---
    np.random.seed(42)
    def loss_func_noise(x, y): return x**2 + y**2
    def grad_func_noise(x, y): 
        noise = np.random.normal(0, 2.0, size=2)
        return np.array([2*x, 2*y]) + noise
        
    start_pos = np.array([-3.0, -3.0])
    iterations = 40
    
    # Adam
    path_adam_noise = [start_pos]
    curr = start_pos.copy()
    m = np.zeros_like(curr); v = np.zeros_like(curr)
    lr = 0.1
    
    for t in range(1, iterations + 1):
        g = grad_func_noise(curr[0], curr[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        curr = curr - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        path_adam_noise.append(curr.copy())
        
    # RMSProp path again
    path_rms_noise = [start_pos]
    curr = start_pos.copy()
    s = np.zeros_like(curr)
    lr_rms = 0.1
    for _ in range(iterations):
        g = grad_func_noise(curr[0], curr[1])
        s = 0.9 * s + 0.1 * g**2
        curr = curr - lr_rms * g / (np.sqrt(s) + 1e-8)
        path_rms_noise.append(curr.copy())

    x = np.linspace(-4, 2, 100); y = np.linspace(-4, 2, 100)
    X, Y = np.meshgrid(x, y); Z = loss_func_noise(X, Y)
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.contour(X, Y, Z, levels=15, colors='#E1D5E7', alpha=0.8)
    
    plot_path(ax2, path_rms_noise, '#9673A6', 'RMSProp (Fails: Jitters)', linestyle='--', marker_size=6)
    plot_path(ax2, path_adam_noise, '#82B366', 'Adam (Success: Smooth)', marker_size=6)
    
    ax2.set_title('Adam vs Noisy Gradients\n(Adam uses momentum to smooth path)', fontsize=11)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adam_vs_noise.png'), dpi=200)
    print("Generated adam_vs_scale.png and adam_vs_noise.png")
    
    # --- Plot 3: RMSProp vs SGD (For A.8.2 benefit) ---
    # User asked: "When talking about RMSProp advantages, compare vs SGD, not Momentum"
    def loss_func_sgd(x, y): return 0.05 * x**2 + 5.0 * y**2
    def grad_func_sgd(x, y): return np.array([0.1 * x, 10.0 * y])
    
    start_pos = np.array([-4.5, 1.0])
    iterations = 50
    
    # SGD - Needs tiny LR
    path_sgd = [start_pos]
    curr = start_pos.copy()
    lr_sgd = 0.02
    for _ in range(iterations):
        g = grad_func_sgd(curr[0], curr[1])
        curr = curr - lr_sgd * g
        path_sgd.append(curr.copy())
        
    # RMSProp
    path_rms_sgd = [start_pos]
    curr = start_pos.copy()
    s = np.zeros_like(curr)
    lr_rms = 0.5 # Can use large LR
    for _ in range(iterations):
        g = grad_func_sgd(curr[0], curr[1])
        s = 0.9 * s + 0.1 * g**2
        curr = curr - lr_rms * g / (np.sqrt(s) + 1e-8)
        path_rms_sgd.append(curr.copy())
        
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-3, 3, 100))
    Z = loss_func_sgd(X, Y)
    ax3.contour(X, Y, Z, levels=np.logspace(-2, 2, 20), colors='#E1D5E7', alpha=0.8)
    ax3.contourf(X, Y, Z, levels=np.logspace(-2, 2, 20), cmap='Greys', alpha=0.1)
    
    plot_path(ax3, path_sgd, '#EA6B66', 'SGD (Slow)', marker_size=6, linestyle='--')
    plot_path(ax3, path_rms_sgd, '#9673A6', 'RMSProp (Fast)', marker_size=6)
    
    ax3.set_title('RMSProp vs SGD\n(Adaptive learning rate handles scale difference)', fontsize=11)
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmsprop_vs_sgd.png'), dpi=200)
    print("Generated rmsprop_vs_sgd.png")

def run_momentum_vs_sgd_a8():
    """
    Scenario: SGD vs Momentum for Appendix A.8
    A cleaner, Morandi-style visualization for A.8 to replace the old sgd_vs_momentum.png
    """
    def loss_func(x, y): return 0.5 * x**2 + 10 * y**2
    def grad_func(x, y): return np.array([x, 20 * y])

    start_pos = np.array([-5.5, 2.5])
    iterations = 50

    # SGD
    path_sgd = [start_pos]
    curr = start_pos.copy()
    lr_sgd = 0.095
    for _ in range(iterations):
        grad = grad_func(curr[0], curr[1])
        curr = curr - lr_sgd * grad
        path_sgd.append(curr.copy())

    # Momentum
    path_mom = [start_pos]
    curr = start_pos.copy()
    v = np.zeros_like(curr)
    beta = 0.9
    lr_mom = 0.01
    for _ in range(iterations):
        grad = grad_func(curr[0], curr[1])
        v = beta * v + grad
        curr = curr - lr_mom * v
        path_mom.append(curr.copy())

    # Plot
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), colors='#E1D5E7', alpha=0.5)
    ax.contourf(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='Greys', alpha=0.1)

    plot_path(ax, path_sgd, '#F8CECC', 'SGD (Oscillation)', marker_size=4)
    plot_path(ax, path_mom, '#6C8EBF', 'Momentum (Inertia)', marker_size=4)

    ax.set_title('Momentum vs SGD\n(Inertia smoothes out oscillation)', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'momentum_vs_sgd_a8.png'), dpi=200)
    print("Generated momentum_vs_sgd_a8.png")

if __name__ == "__main__":
    run_scale_problem_momentum_only()
    run_noise_problem_rmsprop_only()
    run_adam_vs_badcases()
    run_momentum_vs_sgd_a8()

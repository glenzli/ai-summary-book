import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_margin_gamma():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Rotation setup
    theta = np.radians(30)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Common line setup (x=4 in original frame)
    normal = np.array([np.cos(theta), np.sin(theta)])
    point = np.array([4*np.cos(theta), 4*np.sin(theta)]) # A point on the hyperplane
    
    # Plotting range
    x_line = np.linspace(-1, 8, 100)
    
    # Calculate line y coordinates: n_x * x + n_y * y = C
    C = np.dot(normal, point)
    y_line = (C - normal[0] * x_line) / normal[1]

    # --- Subplot 1: Large Gamma (Easy) ---
    ax = axes[0]
    
    # Data points (Original frame: Class 1 x <= 2.5, Class 2 x >= 5.5)
    # Margin distance = 1.5
    X1 = np.array([[1, 2], [2, 3], [2, 1], [2.5, 2]]) 
    X2 = np.array([[5.5, 2], [6, 3], [6, 1], [6.5, 2]])
    
    X1_rot = X1 @ R.T
    X2_rot = X2 @ R.T
    
    ax.scatter(X1_rot[:, 0], X1_rot[:, 1], color='#6C8EBF', s=100, label='Class +1', edgecolor='k', alpha=0.8)
    ax.scatter(X2_rot[:, 0], X2_rot[:, 1], color='#B85450', s=100, label='Class -1', edgecolor='k', alpha=0.8)
    
    ax.plot(x_line, y_line, color='black', linewidth=2, label='Separating Hyperplane')
    
    # Margin boundaries
    # Distance is 1.5. Points on margin are shifted by +/- 1.5 along normal from hyperplane center
    # Ideally, compute constant C for margins.
    # Margin 1 passes through (2.5, 2) rotated.
    # Margin 2 passes through (5.5, 2) rotated.
    p_m1 = np.array([2.5, 2]) @ R.T
    p_m2 = np.array([5.5, 2]) @ R.T
    
    C1 = np.dot(normal, p_m1)
    C2 = np.dot(normal, p_m2)
    
    y_m1 = (C1 - normal[0] * x_line) / normal[1]
    y_m2 = (C2 - normal[0] * x_line) / normal[1]
    
    ax.plot(x_line, y_m1, '--', color='#82B366', linewidth=1.5)
    ax.plot(x_line, y_m2, '--', color='#82B366', linewidth=1.5)
    
    # Annotation for Gamma
    # Draw arrow from hyperplane to margin
    # Center point on hyperplane
    center_original = np.array([4, 2])
    center_rot = center_original @ R.T
    
    # Vector to margin 1 (distance 1.5 in negative x direction original)
    # Normal points to positive x. So -normal * 1.5
    
    ax.annotate('', xy=center_rot, xytext=center_rot - 1.5 * normal,
                arrowprops=dict(arrowstyle='<->', color='#82B366', lw=2))
    ax.text(center_rot[0] - 0.8*normal[0], center_rot[1] - 0.8*normal[1] + 0.3, r'$\gamma_{large}$', 
            fontsize=14, color='#82B366', fontweight='bold', ha='center')

    ax.set_title(r'Large Margin ($\gamma$ large) $\rightarrow$ Easy', fontsize=16)
    ax.axis('equal')
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper left')
    
    # --- Subplot 2: Small Gamma (Hard) ---
    ax = axes[1]
    
    # Data points (Original frame: Class 1 x <= 3.8, Class 2 x >= 4.2)
    # Margin distance = 0.2
    X1_close = np.array([[3.8, 2], [2, 3], [2, 1], [3, 2]]) 
    X2_close = np.array([[4.2, 2], [6, 3], [6, 1], [5, 2]])
    
    X1_rot_c = X1_close @ R.T
    X2_rot_c = X2_close @ R.T
    
    ax.scatter(X1_rot_c[:, 0], X1_rot_c[:, 1], color='#6C8EBF', s=100, edgecolor='k', alpha=0.8)
    ax.scatter(X2_rot_c[:, 0], X2_rot_c[:, 1], color='#B85450', s=100, edgecolor='k', alpha=0.8)
    
    ax.plot(x_line, y_line, color='black', linewidth=2)
    
    p_m1_c = np.array([3.8, 2]) @ R.T
    p_m2_c = np.array([4.2, 2]) @ R.T
    
    C1_c = np.dot(normal, p_m1_c)
    C2_c = np.dot(normal, p_m2_c)
    
    y_m1_c = (C1_c - normal[0] * x_line) / normal[1]
    y_m2_c = (C2_c - normal[0] * x_line) / normal[1]
    
    ax.plot(x_line, y_m1_c, '--', color='#B85450', linewidth=1.5)
    ax.plot(x_line, y_m2_c, '--', color='#B85450', linewidth=1.5)
    
    # Annotation for Gamma
    # Draw arrow from hyperplane to margin (distance 0.2)
    # To make it visible, we might need to zoom in or just point to it.
    
    # Let's draw an arrow slightly offset so it doesn't overlap too much
    offset_pos = center_rot + np.array([0.5, -0.5])
    
    # Arrow for gamma
    ax.annotate('', xy=center_rot, xytext=center_rot - 0.2 * normal,
                arrowprops=dict(arrowstyle='<->', color='#B85450', lw=2))
    
    # Text with pointer because it's small
    ax.annotate(r'$\gamma_{small}$', xy=(center_rot[0] - 0.1*normal[0], center_rot[1] - 0.1*normal[1]), 
                xytext=(center_rot[0] + 1, center_rot[1] - 2),
                arrowprops=dict(arrowstyle='->', color='#B85450', lw=1.5),
                fontsize=14, color='#B85450', fontweight='bold')

    ax.set_title(r'Small Margin ($\gamma$ small) $\rightarrow$ Hard', fontsize=16)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'margin_gamma_comparison.png'), dpi=300)
    print("Generated margin_gamma_comparison.png")

if __name__ == "__main__":
    plot_margin_gamma()

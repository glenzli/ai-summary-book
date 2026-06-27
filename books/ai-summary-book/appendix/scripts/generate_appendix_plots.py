
import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import norm, laplace  <-- Removed dependency
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

# Implement simple PDF functions to avoid scipy dependency
def norm_pdf(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def laplace_pdf(x, mu, b):
    return 1/(2 * b) * np.exp(-np.abs(x - mu)/b)

# Ensure directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../images')
os.makedirs(output_dir, exist_ok=True)

def plot_lagrange_intuition():
    # utils.plot_style is automatically applied
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Grid
    x = np.linspace(-1.5, 2.5, 400)
    y = np.linspace(-1.5, 2.5, 400)
    X, Y = np.meshgrid(x, y)
    
    # Objective Function: f(x,y) = (x-1.2)^2 + (y-1.2)^2
    # Optimum without constraint is at (1.2, 1.2)
    Z = (X - 1.2)**2 + (Y - 1.2)**2
    
    # Constraint: g(x,y) = x^2 + y^2 - 1 = 0
    # Circle
    theta = np.linspace(0, 2*np.pi, 200)
    xc = np.cos(theta)
    yc = np.sin(theta)
    
    # Plot Contours of f
    levels = [0.1, 0.4, 0.9, 1.6, 2.5]
    cs = ax.contour(X, Y, Z, levels=levels, colors='#6C8EBF', alpha=0.6)
    ax.clabel(cs, inline=1, fontsize=10, fmt='f=%.1f')
    
    # Plot Constraint g=0
    ax.plot(xc, yc, color='#B85450', linewidth=3, label='Constraint g(x,y)=0')
    
    # Optimal Point (Tangent)
    # Gradient of f is towards (1.2, 1.2). Gradient of g is outwards from (0,0).
    # They are parallel at intersection of line y=x and circle in first quadrant.
    x_opt = 1/np.sqrt(2)
    y_opt = 1/np.sqrt(2)
    
    ax.scatter([x_opt], [y_opt], color='black', s=100, zorder=10, label='Optimal Solution')
    
    # Draw Gradients
    # Grad f at opt: 2(x-1.2), 2(y-1.2)
    grad_f_x = 2 * (x_opt - 1.2)
    grad_f_y = 2 * (y_opt - 1.2)
    # Normalize for plotting
    norm_f = np.sqrt(grad_f_x**2 + grad_f_y**2)
    grad_f_x /= norm_f
    grad_f_y /= norm_f
    
    # Grad g at opt: 2x, 2y
    grad_g_x = 2 * x_opt
    grad_g_y = 2 * y_opt
    # Normalize
    norm_g = np.sqrt(grad_g_x**2 + grad_g_y**2)
    grad_g_x /= norm_g
    grad_g_y /= norm_g
    
    # Quiver
    ax.arrow(x_opt, y_opt, -grad_f_x*0.5, -grad_f_y*0.5, head_width=0.05, head_length=0.1, fc='#6C8EBF', ec='#6C8EBF', linewidth=2, label='-Grad f (Descent Direction)')
    ax.arrow(x_opt, y_opt, grad_g_x*0.5, grad_g_y*0.5, head_width=0.05, head_length=0.1, fc='#B85450', ec='#B85450', linewidth=2, label='Grad g')
    
    # Annotations
    ax.text(x_opt + 0.1, y_opt + 0.1, 'Tangent Point', fontsize=12, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_title('Geometric Intuition of Lagrange Multipliers\nGradients are Parallel at Optimum', fontsize=14)
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-1.5, 2.0)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'lagrange_geometric.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_priors():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(-4, 4, 1000)
    
    # Gaussian (L2)
    y_gauss = norm_pdf(x, 0, 1)
    # Laplace (L1)
    y_laplace = laplace_pdf(x, 0, 1/np.sqrt(2)) # Match variance for fair comparison
    
    ax.plot(x, y_gauss, label='Gaussian Prior (L2)\n$p(w) \propto e^{-w^2}$', color='#6C8EBF', linewidth=3)
    ax.fill_between(x, y_gauss, alpha=0.2, color='#6C8EBF')
    
    ax.plot(x, y_laplace, label='Laplace Prior (L1)\n$p(w) \propto e^{-|w|}$', color='#D6B656', linewidth=3)
    ax.fill_between(x, y_laplace, alpha=0.2, color='#D6B656')
    
    ax.set_title('Bayesian Priors for Regularization', fontsize=14)
    ax.set_xlabel('Parameter Weight w')
    ax.set_ylabel('Probability Density')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Annotate Peak
    ax.annotate('Sharp Peak at 0\n(Promotes Sparsity)', xy=(0, 0.7), xytext=(1.5, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig(os.path.join(output_dir, 'bayesian_priors.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_bayesian_update():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    w = np.linspace(-2, 6, 1000)
    
    # Prior: Centered at 0 (Small weights preference)
    prior_mean = 0
    prior_std = 1.0
    prior = norm_pdf(w, prior_mean, prior_std)
    
    # Likelihood: Data says w should be around 4
    like_mean = 4
    like_std = 1.0 # Some uncertainty
    likelihood = norm_pdf(w, like_mean, like_std)
    
    # Posterior: Product
    # For Gaussian x Gaussian, posterior is also Gaussian
    # New Mean = (sigma_l^2 * mu_p + sigma_p^2 * mu_l) / (sigma_l^2 + sigma_p^2)
    post_var = 1 / (1/prior_std**2 + 1/like_std**2)
    post_mean = post_var * (prior_mean/prior_std**2 + like_mean/like_std**2)
    post_std = np.sqrt(post_var)
    
    posterior = norm_pdf(w, post_mean, post_std)
    # Scale posterior for visualization (since product is not pdf yet)
    # Actually plotting the correct PDF of posterior is better
    
    ax.plot(w, prior, label='Prior (Regularization)\n"Weights should be small"', color='#D6B656', linestyle='--', linewidth=2)
    ax.plot(w, likelihood, label='Likelihood (Data Loss)\n"Data says w=4"', color='#6C8EBF', linestyle='--', linewidth=2)
    ax.plot(w, posterior, label='Posterior (Result)\nMAP Estimate', color='#B85450', linewidth=3)
    
    ax.fill_between(w, posterior, alpha=0.3, color='#B85450')
    
    # Arrow showing the "Pull"
    ax.annotate('Regularization Pulls\nEstimate towards 0', xy=(post_mean, 0.2), xytext=(1, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax.set_title('How Regularization Affects Parameter Estimation (MAP)', fontsize=14)
    ax.set_xlabel('Parameter Weight w')
    ax.set_yticks([])
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'bayesian_update.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_lagrange_intuition()
    plot_priors()
    plot_bayesian_update()

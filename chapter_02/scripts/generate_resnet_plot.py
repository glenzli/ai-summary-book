import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def create_arrow(ax, start, end, color='#333', style='->', lw=2, curved=False):
    arrow_style = f"{style},head_width=0.3,head_length=0.4"
    conn_style = "arc3,rad=-0.4" if curved else "arc3,rad=0"
    ax.annotate("", xy=end, xytext=start, 
                arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw, connectionstyle=conn_style))

def draw_block(ax, center, width, height, label, color='#DAE8FC', ec='#6C8EBF'):
    x, y = center[0] - width/2, center[1] - height/2
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=ec, facecolor=color)
    ax.add_patch(rect)
    ax.text(center[0], center[1], label, ha='center', va='center', fontsize=12, fontweight='bold')

def plot_resnet_flow():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Coordinates
    x_center = 4
    y_input = 9
    y_bn1 = 7.5
    y_relu1 = 6.5
    y_conv2 = 5
    y_bn2 = 3.5
    y_add = 2
    y_relu2 = 1
    
    # Nodes
    # Input x
    ax.text(x_center, y_input, "$x_l$", ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Weight Layer 1 (Conv + BN)
    draw_block(ax, (x_center, y_bn1), 3, 1, "Weight Layer\n(Conv + BN)")
    
    # ReLU
    draw_block(ax, (x_center, y_relu1), 1.5, 0.6, "ReLU", color='#FFF2CC', ec='#D6B656')
    
    # Weight Layer 2
    draw_block(ax, (x_center, y_conv2), 3, 1, "Weight Layer\n(Conv + BN)")
    
    # Addition Node
    circle = patches.Circle((x_center, y_add), 0.3, facecolor='#F8CECC', edgecolor='#B85450', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_center, y_add, "+", ha='center', va='center', fontsize=20, fontweight='bold', color='#B85450')
    
    # Output ReLU
    draw_block(ax, (x_center, y_relu2), 1.5, 0.6, "ReLU", color='#FFF2CC', ec='#D6B656')
    ax.text(x_center, 0.2, "$x_{l+1}$", ha='center', va='center', fontsize=16, fontweight='bold')
    
    # --- Forward Path Arrows ---
    create_arrow(ax, (x_center, y_input-0.3), (x_center, y_bn1+0.6))
    create_arrow(ax, (x_center, y_bn1-0.6), (x_center, y_relu1+0.4))
    create_arrow(ax, (x_center, y_relu1-0.4), (x_center, y_conv2+0.6))
    create_arrow(ax, (x_center, y_conv2-0.6), (x_center, y_add+0.3))
    create_arrow(ax, (x_center, y_add-0.3), (x_center, y_relu2+0.4))
    create_arrow(ax, (x_center, y_relu2-0.4), (x_center, 0.5))
    
    # --- Skip Connection ---
    # Draw curved line from top to add
    # Using specific path patch for better control than annotate
    path_x = [x_center, x_center + 2.5, x_center + 2.5, x_center + 0.3]
    path_y = [y_input-0.5, y_input-0.5, y_add, y_add]
    
    # Draw skip line
    ax.plot([x_center, x_center+2.5], [y_input-0.5, y_input-0.5], color='#333', lw=2) # Horiz top
    ax.plot([x_center+2.5, x_center+2.5], [y_input-0.5, y_add], color='#333', lw=2) # Vertical
    create_arrow(ax, (x_center+2.5, y_add), (x_center+0.3, y_add), style='->') # Horiz bottom
    
    ax.text(x_center+2.7, (y_input+y_add)/2, "Identity Mapping\n(Skip Connection)", 
            ha='left', va='center', fontsize=12, fontstyle='italic', color='#555')
    
    # --- Gradient Flow Visuals ---
    # Backward arrow on skip connection
    create_arrow(ax, (x_center+2.3, y_add+0.5), (x_center+2.3, y_input-1.0), 
                 color='#B85450', lw=3, style='->')
    ax.text(x_center+2.4, y_bn2+1, "Gradient Superhighway\nNo decay!", 
            ha='left', va='center', fontsize=12, fontweight='bold', color='#B85450')
    
    # Backward arrow on main path (faded)
    create_arrow(ax, (x_center-0.5, y_add+0.5), (x_center-0.5, y_conv2-0.5), 
                 color='#B85450', lw=1.5, style='->')
    ax.text(x_center-0.8, y_bn2, "Decays...", ha='right', va='center', fontsize=10, color='#B85450')

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '../images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'resnet_flow.png'), bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'resnet_flow.png')}")

if __name__ == "__main__":
    plot_resnet_flow()
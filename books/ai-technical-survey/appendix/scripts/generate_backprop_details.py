
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

# Set Morandi colors
colors = {
    'bg': '#FFFFFF',
    'node': '#DAE8FC',       # Blue-ish
    'input': '#F5F5F5',      # Grey-ish
    'output': '#E1D5E7',     # Purple-ish
    'grad_upstream': '#F8CECC', # Red-ish (Upstream)
    'grad_local': '#FFF2CC',    # Yellow-ish (Local)
    'text': '#333333',
    'arrow_fwd': '#6C8EBF',
    'arrow_bwd': '#B85450',
    'highlight': '#333333'      # Changed from red to dark grey
}

def create_figure(figsize=(10, 6)):
    # utils.plot_style is automatically applied
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(colors['bg'])
    ax.set_facecolor(colors['bg'])
    ax.axis('off')
    ax.set_aspect('equal') # Ensure circles are round
    return fig, ax

def draw_arrow(ax, start, end, color=colors['arrow_fwd'], style='->', lw=3, curved=False):
    arrow_style = f"{style},head_width=0.4,head_length=0.5" # Larger arrow heads
    connection_style = "arc3,rad=0.2" if curved else "arc3,rad=0"
    ax.annotate("", xy=end, xytext=start, 
                arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw, connectionstyle=connection_style))

def plot_node_view():
    # Reduced width/height to make content fill the space better (less margin)
    fig, ax = create_figure(figsize=(12, 6))
    
    # Coordinates (Slightly compressed to reduce whitespace)
    x_node, y_node = 6, 5
    x_in1, y_in1 = 2.5, 7.5
    x_in2, y_in2 = 2.5, 2.5
    x_out, y_out = 9.5, 5
    
    # 1. Draw Forward Path (Subtle)
    # Inputs (Larger circles)
    c1 = patches.Circle((x_in1, y_in1), 0.6, fc=colors['input'], ec='#CCCCCC', zorder=2)
    c2 = patches.Circle((x_in2, y_in2), 0.6, fc=colors['input'], ec='#CCCCCC', zorder=2)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(x_in1, y_in1, "$x$", ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(x_in2, y_in2, "$y$", ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Node (Smaller box, tighter content)
    # Reduced size to make content occupy more relative space
    box = patches.FancyBboxPatch((x_node-1.0, y_node-1.0), 2.0, 2.0, boxstyle="round,pad=0.05", 
                               fc=colors['node'], ec='#6C8EBF', zorder=2, lw=2)
    ax.add_patch(box)
    
    # Content inside Node - Expanded
    ax.text(x_node, y_node, "$f(x,y)$", ha='center', va='center', fontsize=22, fontweight='bold', color='#333')
    
    # Output (Larger circle)
    c_out = patches.Circle((x_out, y_out), 0.6, fc=colors['output'], ec='#CCCCCC', zorder=2)
    ax.add_patch(c_out)
    ax.text(x_out, y_out, "$z$", ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Forward Edges (Thicker)
    draw_arrow(ax, (x_in1+0.6, y_in1-0.2), (x_node-1.0, y_node+0.5), color='#DDDDDD', lw=4) # Adjusted connection points
    draw_arrow(ax, (x_in2+0.6, y_in2+0.2), (x_node-1.0, y_node-0.5), color='#DDDDDD', lw=4)
    draw_arrow(ax, (x_node+1.0, y_node), (x_out-0.6, y_out), color='#DDDDDD', lw=4)
    
    # 2. Draw Backward Path (Prominent & Larger Text)
    # Upstream Gradient
    draw_arrow(ax, (x_out+2.2, y_out+1.5), (x_out, y_out+0.8), color=colors['arrow_bwd'], lw=3.5)
    # Significantly larger Upstream Gradient
    # Split text into label (smaller) and math (larger)
    ax.text(x_out+2.4, y_out+2.2, "Upstream Gradient", 
            ha='center', va='bottom', fontsize=14, color=colors['arrow_bwd'], fontweight='bold')
    ax.text(x_out+2.4, y_out+1.5, r"$\frac{\partial L}{\partial z}$", 
            ha='center', va='bottom', fontsize=22, color=colors['arrow_bwd'], fontweight='bold')
    
    # Local Gradient (Inside Node)
    # Moved text to be clearer
    ax.text(x_node, y_node+1.5, "Local Gradient", ha='center', va='bottom', fontsize=14, color='#555', fontweight='bold')
    ax.text(x_node-0.5, y_node+0.6, r"$\frac{\partial z}{\partial x}$", fontsize=20, color='#333')
    ax.text(x_node-0.5, y_node-0.6, r"$\frac{\partial z}{\partial y}$", fontsize=20, color='#333')
    
    # Downstream Gradients (Result - Big & Bold)
    # To x
    draw_arrow(ax, (x_node-1.1, y_node+0.5), (x_in1+0.9, y_in1), color=colors['arrow_bwd'], lw=3.5)
    ax.text(3.5, 7.8, r"$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$", 
            ha='center', va='bottom', fontsize=22, color=colors['arrow_bwd'], fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='#FFF', ec=colors['arrow_bwd'], lw=2))
            
    # To y
    draw_arrow(ax, (x_node-1.1, y_node-0.5), (x_in2+0.9, y_in2), color=colors['arrow_bwd'], lw=3.5)
    ax.text(3.5, 2.2, r"$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y}$", 
            ha='center', va='top', fontsize=22, color=colors['arrow_bwd'], fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='#FFF', ec=colors['arrow_bwd'], lw=2))

    ax.set_xlim(1, 13)
    ax.set_ylim(1, 9)
    plt.tight_layout()
    plt.savefig('appendix/images/backprop_node.png', bbox_inches='tight')

def plot_gates_view():
    fig, ax = create_figure(figsize=(16, 5)) # Reduced height from 6 to 5
    
    # Positions - Increased spacing
    gates = [
        {"name": "ADD Gate", "op": "+", "x": 2.5, "desc": "Gradient Distributor", "rule": "x.grad = 1.0 * grad\ny.grad = 1.0 * grad"},
        {"name": "MUL Gate", "op": r"$\times$", "x": 8.0, "desc": "Gradient Switcher", "rule": "x.grad = y * grad\ny.grad = x * grad"},
        {"name": "ReLU Gate", "op": "max", "x": 13.5, "desc": "Gradient Filter", "rule": "If x > 0: grad\nIf x <= 0: 0"}
    ]
    
    y_center = 3.2 # Moved down slightly to fit tight layout
    
    for gate in gates:
        gx, gy = gate["x"], y_center
        
        # Draw Gate Node (Smaller Size)
        circle = patches.Circle((gx, gy), 0.5, fc=colors['node'], ec='#6C8EBF', lw=2.5)
        ax.add_patch(circle)
        
        # Op symbol
        ax.text(gx, gy, gate["op"], ha='center', va='center', fontsize=16, fontweight='bold', color='#333')
        # Name
        ax.text(gx, gy+0.8, gate["name"], ha='center', va='bottom', fontsize=18, fontweight='bold', color='#333')
        
        # Upstream Gradient (Incoming from right - Shorter)
        # Length reduced to 1.5 (end at gx+2.1, start at gx+0.6)
        draw_arrow(ax, (gx+2.1, gy), (gx+0.6, gy), color=colors['arrow_bwd'], lw=4)
        # "grad" text at the tail
        ax.text(gx+2.2, gy, "grad", ha='left', va='center', fontsize=16, color=colors['arrow_bwd'], fontweight='bold')
        # "z" label near the gate port
        ax.text(gx+0.7, gy+0.2, "z", ha='left', va='bottom', fontsize=14, color='#333', fontweight='bold')
        
        # Inputs (Backward flow out - Shorter)
        # Length reduced to 1.5
        if gate["name"] == "ReLU Gate":
            draw_arrow(ax, (gx-0.6, gy), (gx-2.1, gy), color=colors['arrow_bwd'], lw=4)
            # "x" label near the gate port
            ax.text(gx-0.7, gy+0.2, "x", ha='right', va='bottom', fontsize=14, color='#333', fontweight='bold')
        else:
            # Top arrow (x)
            draw_arrow(ax, (gx-0.5, gy+0.2), (gx-2.0, gy+0.8), color=colors['arrow_bwd'], lw=4)
            ax.text(gx-0.6, gy+0.3, "x", ha='right', va='bottom', fontsize=14, color='#333', fontweight='bold')
            
            # Bottom arrow (y)
            draw_arrow(ax, (gx-0.5, gy-0.2), (gx-2.0, gy-0.8), color=colors['arrow_bwd'], lw=4)
            ax.text(gx-0.6, gy-0.3, "y", ha='right', va='top', fontsize=14, color='#333', fontweight='bold')
        
        # Description Text (Larger Font)
        ax.text(gx, gy-1.0, f"{gate['desc']}", ha='center', va='top', fontweight='bold', fontsize=16, color='#333')
        ax.text(gx, gy-1.5, gate['rule'], ha='center', va='top', fontsize=14, color='#555', linespacing=1.3)

    ax.set_xlim(0, 16)
    # Tighter vertical crop: content is roughly from 1.5 to 4.5
    ax.set_ylim(1.0, 5.0)
    plt.tight_layout()
    
    # Ensure directory exists using absolute path
    output_dir = os.path.join(os.path.dirname(__file__), '../images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'backprop_gates.png'), bbox_inches='tight')

if __name__ == "__main__":
    plot_node_view()
    plot_gates_view()

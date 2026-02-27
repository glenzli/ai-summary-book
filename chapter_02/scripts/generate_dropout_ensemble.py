
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set Morandi colors
colors = {
    'bg': '#FAFAFA',
    'neuron_active': '#DAE8FC',  # Blue
    'neuron_drop': '#F5F5F5',    # Grey
    'neuron_input': '#D5E8D4',   # Green
    'neuron_output': '#E1D5E7',  # Purple
    'edge_active': '#6C8EBF',
    'edge_drop': '#DDDDDD',
    'text': '#333333'
}

def draw_network(ax, layer_sizes, active_nodes=None, title="", offset=(0, 0), scale=1.0, show_edges=True):
    """
    Draws a neural network.
    active_nodes: list of sets, where each set contains indices of active nodes for that layer.
                  If None, all nodes are active.
    """
    v_spacing = 0.8 * scale
    h_spacing = 1.2 * scale # Slightly tighter horizontal spacing
    
    # Calculate positions
    layers = []
    max_nodes = max(layer_sizes)
    
    for i, size in enumerate(layer_sizes):
        layer_nodes = []
        layer_height = (size - 1) * v_spacing
        start_y = -layer_height / 2 + offset[1]
        x = i * h_spacing + offset[0]
        
        for j in range(size):
            y = start_y + j * v_spacing
            layer_nodes.append((x, y))
        layers.append(layer_nodes)
    
    # Draw edges
    if show_edges:
        for i in range(len(layers) - 1):
            curr_layer = layers[i]
            next_layer = layers[i+1]
            
            for j, (x1, y1) in enumerate(curr_layer):
                for k, (x2, y2) in enumerate(next_layer):
                    # Check if nodes are active
                    active_u = (active_nodes is None) or (j in active_nodes[i])
                    active_v = (active_nodes is None) or (k in active_nodes[i+1])
                    
                    if active_u and active_v:
                        color = colors['edge_active']
                        alpha = 0.8
                        lw = 1.5 * scale
                        zorder = 1
                    else:
                        color = colors['edge_drop']
                        alpha = 0.3
                        lw = 1.0 * scale
                        zorder = 0
                        
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=lw, zorder=zorder)

    # Draw nodes
    for i, layer in enumerate(layers):
        for j, (x, y) in enumerate(layer):
            # Determine status
            is_active = (active_nodes is None) or (j in active_nodes[i])
            
            if i == 0: # Input
                fc = colors['neuron_input'] if is_active else colors['neuron_drop']
                ec = '#82B366' if is_active else '#CCCCCC'
            elif i == len(layers) - 1: # Output
                fc = colors['neuron_output'] if is_active else colors['neuron_drop']
                ec = '#9673A6' if is_active else '#CCCCCC'
            else: # Hidden
                fc = colors['neuron_active'] if is_active else colors['neuron_drop']
                ec = '#6C8EBF' if is_active else '#CCCCCC'
            
            if not is_active:
                ec = '#CCCCCC' # Force grey border for dropped
            
            circle = patches.Circle((x, y), radius=0.12*scale, facecolor=fc, edgecolor=ec, lw=1.5*scale, zorder=2)
            ax.add_patch(circle)
            
            # Cross out dropped nodes
            if not is_active:
                r = 0.08 * scale
                ax.plot([x-r, x+r], [y-r, y+r], color='#999999', lw=1*scale, zorder=3)
                ax.plot([x-r, x+r], [y+r, y-r], color='#999999', lw=1*scale, zorder=3)

    # Title
    # Center title over the network
    net_width = (len(layer_sizes)-1) * h_spacing
    center_x = offset[0] + net_width / 2
    # Place title slightly higher
    title_y = offset[1] + (max_nodes * v_spacing)/2 + 0.6 * scale
    
    ax.text(center_x, title_y, title, ha='center', va='bottom', fontsize=10*scale, fontweight='bold', color=colors['text'])

def generate_dropout_plot():
    # Increase figure width to accommodate wider xlim and preserve aspect ratio
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor(colors['bg'])
    fig.patch.set_facecolor(colors['bg'])
    
    # Force equal aspect ratio to prevent stretching
    ax.set_aspect('equal')
    
    layer_sizes = [3, 4, 4, 2]
    net_scale = 0.9
    net_width_approx = 3.5 # Roughly (len-1) * h_spacing
    
    # Layout positions
    y_train = 3.5
    y_test = -3.5 # Move down significantly to avoid overlap
    x_gap = 4.5
    
    # --- Sub-network 1 ---
    active_1 = [
        set([0, 1, 2]),      # Input (usually kept)
        set([0, 2, 3]),      # Hidden 1 (Drop index 1)
        set([1, 2]),         # Hidden 2 (Drop index 0, 3)
        set([0, 1])          # Output
    ]
    draw_network(ax, layer_sizes, active_1, title="Training: Sub-network A", offset=(0, y_train), scale=net_scale)

    # --- Sub-network 2 ---
    active_2 = [
        set([0, 1, 2]),
        set([1, 2]),         # Hidden 1
        set([0, 3]),         # Hidden 2
        set([0, 1])
    ]
    draw_network(ax, layer_sizes, active_2, title="Training: Sub-network B", offset=(x_gap, y_train), scale=net_scale)
    
    # --- Sub-network 3 ---
    active_3 = [
        set([0, 1, 2]),
        set([0, 3]),
        set([1, 2, 3]),
        set([0, 1])
    ]
    draw_network(ax, layer_sizes, active_3, title="Training: Sub-network C", offset=(x_gap*2, y_train), scale=net_scale)

    # --- Ensemble / Testing ---
    # Center the test network under the middle training network
    draw_network(ax, layer_sizes, None, title="Testing: Full Network (Scaled Weights)", offset=(x_gap, y_test), scale=net_scale)
    
    # Add annotations
    
    # Centers of networks (approximate)
    net_w = (len(layer_sizes)-1) * (1.2 * net_scale)
    c1_x = 0 + net_w/2
    c2_x = x_gap + net_w/2
    
    # Main Title
    # Use c2_x as the center since it's the middle network
    ax.text(c2_x, y_train + 2.8, "Ensemble Learning View: Training $2^N$ Sub-networks", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
            
    # Arrows
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="#666666")
    c3_x = x_gap*2 + net_w/2
    ct_x = x_gap + net_w/2
    
    bottom_train = y_train - 1.5
    top_test = -1.0 # Arrow points to y=-1.0, giving space before the title at approx -1.5
    
    # A to Test
    a1 = patches.FancyArrowPatch((c1_x, bottom_train), (ct_x - 1, top_test), connectionstyle="arc3,rad=0.2", **kw)
    ax.add_patch(a1)
    
    # B to Test
    a2 = patches.FancyArrowPatch((c2_x, bottom_train), (ct_x, top_test), connectionstyle="arc3,rad=0", **kw)
    ax.add_patch(a2)
    
    # C to Test
    a3 = patches.FancyArrowPatch((c3_x, bottom_train), (ct_x + 1, top_test), connectionstyle="arc3,rad=-0.2", **kw)
    ax.add_patch(a3)
    
    # Text position
    text_y = (bottom_train + top_test)/2
    ax.text(c2_x, text_y, "Approximated by\nWeight Scaling", ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF2CC", ec="#D6B656", alpha=0.8))

    # Adjust limits to ensure everything is visible
    # Leftmost x is 0. Rightmost x is x_gap*2 + net_w ~ 9 + 3.2 = 12.2
    # User requested tighter margins on sides and top/bottom
    ax.set_xlim(-1.5, 14.0)
    ax.set_ylim(-5.5, 7.5) # Increased top to prevent title overlap
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('chapter_02/images/dropout_ensemble.png', bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    generate_dropout_plot()

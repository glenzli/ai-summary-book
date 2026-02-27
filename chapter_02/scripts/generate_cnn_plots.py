import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def draw_grid(ax, rows, cols, offset=(0,0), cell_size=1.0, color='#DAE8FC', label=None, linewidth=1.5):
    x_off, y_off = offset
    
    # Draw background rectangle
    rect = patches.Rectangle((x_off, y_off), cols*cell_size, rows*cell_size, 
                             linewidth=linewidth, edgecolor='#6C8EBF', facecolor=color, alpha=0.3)
    ax.add_patch(rect)
    
    # Draw grid lines
    for r in range(rows + 1):
        ax.plot([x_off, x_off + cols*cell_size], [y_off + r*cell_size, y_off + r*cell_size], color='#6C8EBF', lw=linewidth*0.5)
    for c in range(cols + 1):
        ax.plot([x_off + c*cell_size, x_off + c*cell_size], [y_off, y_off + rows*cell_size], color='#6C8EBF', lw=linewidth*0.5)
        
    if label:
        ax.text(x_off + cols*cell_size/2, y_off - 0.5, label, ha='center', va='top', fontsize=12, fontweight='bold')

def highlight_cell(ax, r, c, rows, cols, offset=(0,0), cell_size=1.0, color='#FFD966', alpha=0.8):
    x_off, y_off = offset
    # Grid coordinates start from bottom-left
    # r=0 is bottom row
    rect = patches.Rectangle((x_off + c*cell_size, y_off + (rows-1-r)*cell_size), cell_size, cell_size, 
                             linewidth=2, edgecolor='#D6B656', facecolor=color, alpha=alpha)
    ax.add_patch(rect)

def connect_patches(ax, r_in, c_in, size_in, r_out, c_out, offset_in, offset_out, cell_size=1.0):
    # Connect a region in input to a pixel in output
    x_in, y_in = offset_in
    x_out, y_out = offset_out
    
    # Input region corners (top-left, top-right, bottom-left, bottom-right)
    # y coordinate for row r is y_off + (rows-1-r)*cell_size
    # Top-left of the region
    
    # Input box (3x3)
    # Top-left of input window
    x1 = x_in + c_in * cell_size
    y1 = y_in + (size_in[0] - r_in) * cell_size 
    
    # Bottom-right of input window
    # window size is 3x3 usually
    k = 3
    x2 = x_in + (c_in + k) * cell_size
    y2 = y_in + (size_in[0] - r_in - k) * cell_size
    
    # Output pixel center
    x_o = x_out + (c_out + 0.5) * cell_size
    y_o = y_out + (size_in[0] - 2 - r_out - 0.5) * cell_size # roughly matching height
    
    # Draw lines
    con_color = '#B85450'
    alpha = 0.4
    
    # Connect corners to center
    ax.plot([x1, x_o], [y1, y_o], color=con_color, alpha=alpha, linestyle='--')
    ax.plot([x2, y1], [x_o, y_o], color=con_color, alpha=alpha, linestyle='--') # Wait, coordinates mixed
    
    # Let's simplify: draw a pyramid
    poly_pts = [
        [x1, y1], # Top Left of Kernel
        [x2, y1], # Top Right of Kernel
        [x2, y2], # Bottom Right of Kernel
        [x1, y2]  # Bottom Left of Kernel
    ]
    
    # Create a polygon representing the receptive field cone
    # Ideally we want a 3D effect, but 2D is fine.
    # Just draw lines from the 4 corners of the kernel window to the output pixel
    
    # Output pixel position
    # Let's assume output grid is to the right
    # Adjust y_o calculation based on output grid params passed or inferred
    pass

def plot_convolution_diagram():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input Grid (5x5)
    draw_grid(ax, 5, 5, offset=(1, 2), label="Input Image\n(Padding=1)", color='#F5F5F5')
    
    # Padding border
    # Highlight inner 3x3 as original image, outer ring as padding
    rect = patches.Rectangle((1+1, 2+1), 3, 3, linewidth=1, edgecolor='#999', fill=False, linestyle='--')
    ax.add_patch(rect)
    ax.text(2.5, 5.5, "Original", fontsize=10, color='#666', ha='center')
    
    # Kernel Window (3x3) at position (0,0) of input (which includes padding)
    # Let's say we are convolving the top-left corner
    k_x, k_y = 1, 2 + 2 # Position corresponding to top-left 3x3 in the 5x5 grid
    
    # Highlight the current 3x3 window on Input
    # 5 rows. Top row is row 0. 
    # highlight_cell logic: r=0 is top visual row? No, r=0 is bottom in previous logic.
    # Let's fix highlight_cell to be intuitive: r=0 is top row.
    
    # Redefine draw_grid for top-down coordinates
    # Clear ax to restart
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.invert_yaxis() # Top-down
    
    def draw_grid_topdown(ax, rows, cols, offset=(0,0), cell_size=1.0, color='#DAE8FC', label=None):
        x_off, y_off = offset
        rect = patches.Rectangle((x_off, y_off), cols*cell_size, rows*cell_size, 
                                 linewidth=1.5, edgecolor='#6C8EBF', facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        for r in range(rows + 1):
            ax.plot([x_off, x_off + cols*cell_size], [y_off + r*cell_size, y_off + r*cell_size], color='#6C8EBF', lw=0.5)
        for c in range(cols + 1):
            ax.plot([x_off + c*cell_size, x_off + c*cell_size], [y_off, y_off + rows*cell_size], color='#6C8EBF', lw=0.5)
        if label:
            ax.text(x_off + cols*cell_size/2, y_off + rows*cell_size + 0.5, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def highlight_region(ax, r, c, h, w, offset=(0,0), cell_size=1.0, color='#FFD966', label=None):
        x_off, y_off = offset
        rect = patches.Rectangle((x_off + c*cell_size, y_off + r*cell_size), w*cell_size, h*cell_size, 
                                 linewidth=2, edgecolor='#D6B656', facecolor=color, alpha=0.5)
        ax.add_patch(rect)
        if label:
             ax.text(x_off + (c + w/2)*cell_size, y_off + r*cell_size - 0.2, label, ha='center', va='bottom', fontsize=10, color='#B85450', fontweight='bold')

    # 1. Input Feature Map (5x5)
    in_off = (1, 2)
    draw_grid_topdown(ax, 5, 5, offset=in_off, label="Input Feature Map\n(5x5, Pad=1)", color='#F5F5F5')
    
    # Dashed line for padding (inner 3x3)
    rect = patches.Rectangle((in_off[0]+1, in_off[1]+1), 3, 3, linewidth=1.5, edgecolor='#999', fill=False, linestyle='--')
    ax.add_patch(rect)
    
    # 2. Kernel (3x3) visualization - Floating between input and output
    # Instead of floating, let's just show the window on Input
    
    # 3. Output Feature Map (3x3)
    # Formula: (5 - 3 + 0)/1 + 1 = 3
    out_off = (10, 3)
    draw_grid_topdown(ax, 3, 3, offset=out_off, label="Output Feature Map\n(3x3)", color='#E1D5E7')
    
    # --- Animation Frame 1: Calculating output(0,0) ---
    # Highlight Input region (0,0) to (2,2)
    highlight_region(ax, 0, 0, 3, 3, offset=in_off, label="Receptive Field")
    
    # Highlight Output pixel (0,0)
    highlight_region(ax, 0, 0, 1, 1, offset=out_off, color='#B85450', label="Activation")
    
    # Draw Lines connecting
    # From corners of Input Window to corners of Output Pixel
    con_kwargs = dict(color='#B85450', alpha=0.3, linestyle='-')
    ax.plot([in_off[0]+3, out_off[0]], [in_off[1], out_off[1]], **con_kwargs) # Top Right -> Top Left
    ax.plot([in_off[0]+3, out_off[0]], [in_off[1]+3, out_off[1]+1], **con_kwargs) # Bottom Right -> Bottom Left
    
    # Add Kernel Weights Text
    # Draw a small kernel matrix symbol
    ax.text(7.5, 3.5, r"$\ast$", fontsize=40, ha='center', va='center', color='#333')
    ax.text(7.5, 4.5, "Kernel $K$\n(3x3)", fontsize=12, ha='center', va='top', color='#333')
    
    # Annotations
    ax.text(8, 7, r"Output Size = $\lfloor \frac{H + 2P - K}{S} \rfloor + 1$", 
            fontsize=12, ha='center', bbox=dict(facecolor='white', edgecolor='#CCC', pad=10))

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '../images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cnn_spatial.png'), bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'cnn_spatial.png')}")

if __name__ == "__main__":
    plot_convolution_diagram()
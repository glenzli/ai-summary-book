import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def create_feature_vis():
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Define styles
    titles = ['Layer 1\n(Edges & Colors)', 'Layer 3\n(Textures & Patterns)', 'Layer 5\n(Parts & Objects)']
    
    # 1. Layer 1: Edges (Gabor-like)
    ax1 = plt.subplot(131)
    ax1.set_title(titles[0], fontsize=14, pad=20)
    ax1.axis('off')
    
    # Draw Gabor-like filters
    for i in range(4):
        for j in range(4):
            x = i * 30 + 15
            y = j * 30 + 15
            angle = np.random.rand() * 180
            # Draw line segments to simulate edges
            dx = 10 * np.cos(np.deg2rad(angle))
            dy = 10 * np.sin(np.deg2rad(angle))
            
            # Simple edges
            color = plt.cm.gray(np.random.rand()) if np.random.rand() > 0.5 else plt.cm.jet(np.random.rand())
            ax1.plot([x-dx, x+dx], [y-dy, y+dy], color=color, linewidth=2)
            
            # Background square
            rect = Rectangle((x-12, y-12), 24, 24, fill=False, color='#CCCCCC', alpha=0.5)
            ax1.add_patch(rect)
    
    ax1.set_xlim(0, 130)
    ax1.set_ylim(0, 130)
    ax1.invert_yaxis()

    # 2. Layer 3: Textures (Simple Shapes)
    ax2 = plt.subplot(132)
    ax2.set_title(titles[1], fontsize=14, pad=20)
    ax2.axis('off')
    
    # Draw simple shapes/textures
    for i in range(3):
        for j in range(3):
            x = i * 40 + 20
            y = j * 40 + 20
            
            type_idx = np.random.randint(0, 3)
            if type_idx == 0: # Circles (eyes/wheels?)
                circle = plt.Circle((x, y), 12, color='#6C8EBF', alpha=0.7)
                ax2.add_patch(circle)
            elif type_idx == 1: # Crosses/Corners
                ax2.plot([x-10, x+10], [y, y], color='#B85450', linewidth=3)
                ax2.plot([x, x], [y-10, y+10], color='#B85450', linewidth=3)
            else: # Parallel lines (grids)
                ax2.plot([x-10, x+10], [y-5, y-5], color='#82B366', linewidth=2)
                ax2.plot([x-10, x+10], [y+5, y+5], color='#82B366', linewidth=2)
                
            rect = Rectangle((x-18, y-18), 36, 36, fill=False, color='#CCCCCC', alpha=0.5)
            ax2.add_patch(rect)
            
    ax2.set_xlim(0, 130)
    ax2.set_ylim(0, 130)
    ax2.invert_yaxis()

    # 3. Layer 5: Objects (Simplified Representations)
    ax3 = plt.subplot(133)
    ax3.set_title(titles[2], fontsize=14, pad=20)
    ax3.axis('off')
    
    # Draw simplified objects
    # Face-like
    x, y = 30, 30
    face = plt.Circle((x, y), 15, color='#FFE6CC')
    eye1 = plt.Circle((x-5, y-5), 3, color='black')
    eye2 = plt.Circle((x+5, y-5), 3, color='black')
    mouth = Rectangle((x-5, y+5), 10, 3, color='#CC0000')
    ax3.add_patch(face); ax3.add_patch(eye1); ax3.add_patch(eye2); ax3.add_patch(mouth)
    
    # Car-like
    x, y = 90, 30
    body = Rectangle((x-20, y-10), 40, 15, color='#DAE8FC')
    top = Rectangle((x-10, y-20), 20, 10, color='#DAE8FC')
    w1 = plt.Circle((x-10, y+5), 5, color='#333333')
    w2 = plt.Circle((x+10, y+5), 5, color='#333333')
    ax3.add_patch(body); ax3.add_patch(top); ax3.add_patch(w1); ax3.add_patch(w2)
    
    # House-like
    x, y = 60, 90
    house = Rectangle((x-15, y-10), 30, 20, color='#D5E8D4')
    roof = plt.Polygon([[x-20, y-10], [x+20, y-10], [x, y-30]], color='#B85450')
    ax3.add_patch(house); ax3.add_patch(roof)
    
    ax3.set_xlim(0, 130)
    ax3.set_ylim(0, 130)
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.savefig('chapter_02/images/cnn_feature_hierarchy.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_feature_vis()

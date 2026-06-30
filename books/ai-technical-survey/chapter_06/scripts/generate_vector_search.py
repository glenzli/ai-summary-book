import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def plot_vector_search():
    # Setup styling
    # utils.plot_style is automatically applied
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Generate hypothetical 2D embeddings
    np.random.seed(42)
    
    # Cluster 1: "Apple" context (Fruit)
    fruit_cluster = np.random.normal(loc=[2, 2], scale=0.8, size=(20, 2))
    # Cluster 2: "Apple" context (Tech)
    tech_cluster = np.random.normal(loc=[6, 6], scale=0.8, size=(20, 2))
    # Cluster 3: Unrelated (Cars)
    car_cluster = np.random.normal(loc=[2, 7], scale=0.8, size=(15, 2))
    
    # Query: "iPhone battery life"
    query_vec = np.array([5.5, 5.5])
    
    # Plot clusters
    ax.scatter(fruit_cluster[:, 0], fruit_cluster[:, 1], c='#82B366', label='Docs: Fruit/Food', alpha=0.6, s=100, edgecolors='w')
    ax.scatter(tech_cluster[:, 0], tech_cluster[:, 1], c='#6C8EBF', label='Docs: Technology', alpha=0.6, s=100, edgecolors='w')
    ax.scatter(car_cluster[:, 0], car_cluster[:, 1], c='#E1D5E7', label='Docs: Automotive', alpha=0.6, s=100, edgecolors='w')
    
    # Plot Query
    ax.scatter(query_vec[0], query_vec[1], c='#B85450', label='Query: "iPhone battery"', s=200, marker='*', zorder=10, edgecolors='w')
    
    # Draw arrow to nearest neighbors
    # Find 3 nearest in tech cluster
    dists = np.linalg.norm(tech_cluster - query_vec, axis=1)
    nearest_indices = np.argsort(dists)[:3]
    
    for idx in nearest_indices:
        target = tech_cluster[idx]
        ax.annotate("", xy=target, xytext=query_vec, arrowprops=dict(arrowstyle="->", color='#B85450', ls='--'))
        # Circle the match
        circle = plt.Circle(target, 0.3, color='#B85450', fill=False, lw=2)
        ax.add_patch(circle)

    # Annotations and Labels
    ax.set_title("Vector Search: Retrieval Augmented Generation (RAG)", fontsize=16, pad=20)
    ax.set_xlabel("Embedding Dimension 1", fontsize=12)
    ax.set_ylabel("Embedding Dimension 2", fontsize=12)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Grid and limits
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)

    # Explanation text
    text_str = "Embedding Model maps semantics to distance.\nClosest docs (blue) are retrieved for the query (red)."
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.5, 0.5, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    output_path = 'chapter_06/images/vector_search_plot.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_vector_search()

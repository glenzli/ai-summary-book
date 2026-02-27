import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    # Simple patchify visualization (illustrative, not tied to a real image).
    H, W = 224, 224
    patch = 16
    n_h, n_w = H // patch, W // patch

    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    # Background "image"
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="#F5F5F5", edgecolor="#666666", linewidth=1.5))

    # Patch grid
    for i in range(n_h):
        for j in range(n_w):
            x, y = j * patch, H - (i + 1) * patch
            ax.add_patch(
                Rectangle(
                    (x, y),
                    patch,
                    patch,
                    facecolor="#DAE8FC" if (i + j) % 2 == 0 else "#D5E8D4",
                    edgecolor="#FFFFFF",
                    linewidth=0.6,
                    alpha=0.9,
                )
            )

    # Highlight a few patches
    highlights = [(2, 3), (6, 8), (10, 1)]
    for (i, j) in highlights:
        x, y = j * patch, H - (i + 1) * patch
        ax.add_patch(Rectangle((x, y), patch, patch, fill=False, edgecolor="#B85450", linewidth=2.0))

    ax.set_xlim(-10, W + 140)
    ax.set_ylim(-10, H + 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Arrow and token sequence sketch
    ax.annotate(
        "",
        xy=(W + 120, H / 2),
        xytext=(W + 10, H / 2),
        arrowprops=dict(arrowstyle="->", color="#6C8EBF", linewidth=2.2),
    )

    ax.text(W + 18, H / 2 + 22, "Flatten + Linear", color="#6C8EBF", fontsize=11, weight="bold")

    token_y = H / 2 - 35
    token_x0 = W + 30
    for k in range(6):
        ax.add_patch(Rectangle((token_x0 + k * 18, token_y), 14, 14, facecolor="#FFF2CC", edgecolor="#D6B656", linewidth=1.0))
    ax.text(token_x0, token_y - 18, "patch tokens", fontsize=10)

    ax.set_title("ViT Patchify: image → patch tokens (illustrative)")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "vit_patchify_demo.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def make_toy_alignment():
    """Create a toy attention alignment matrix alpha[t, j]. Rows sum to 1."""
    src_tokens = ["I", "like", "apples", "."]
    # Use ASCII tokens to avoid font-glyph issues across renderers.
    tgt_tokens = ["wo", "xihuan", "pingguo", "."]

    # A near-diagonal alignment with small off-diagonal mass.
    alpha = np.array(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.10, 0.75, 0.12, 0.03],
            [0.05, 0.15, 0.75, 0.05],
            [0.02, 0.05, 0.18, 0.75],
        ],
        dtype=float,
    )

    alpha = alpha / alpha.sum(axis=1, keepdims=True)
    return src_tokens, tgt_tokens, alpha


def main():
    src_tokens, tgt_tokens, alpha = make_toy_alignment()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(alpha, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_title("Attention Alignment Matrix (Toy Example)")
    ax.set_xlabel("Source tokens (j)")
    ax.set_ylabel("Target tokens (t)")

    ax.set_xticks(np.arange(len(src_tokens)))
    ax.set_xticklabels(src_tokens)
    ax.set_yticks(np.arange(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)

    # Light annotation for readability
    for t in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            val = alpha[t, j]
            ax.text(
                j,
                t,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#000000" if val < 0.65 else "#FFFFFF",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight α")

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "images",
        "attention_alignment_heatmap.png",
    )
    out_path = os.path.abspath(out_path)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    n = 12

    # 1 = visible, 0 = masked
    bert = np.ones((n, n), dtype=float)
    gpt = np.tril(np.ones((n, n), dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    im0 = axes[0].imshow(bert, cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("BERT (Encoder): bidirectional visibility")
    axes[0].set_xlabel("Key position j")
    axes[0].set_ylabel("Query position i")

    im1 = axes[1].imshow(gpt, cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("GPT (Decoder): causal visibility")
    axes[1].set_xlabel("Key position j")
    axes[1].set_ylabel("Query position i")

    for ax in axes:
        ax.set_xticks([0, n // 2, n - 1])
        ax.set_yticks([0, n // 2, n - 1])

    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Visible (1) vs Masked (0)")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "bert_vs_gpt_attention_mask.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

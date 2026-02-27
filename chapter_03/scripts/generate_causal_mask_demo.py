import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def main():
    n = 10

    # Causal mask: allow j <= i.
    mask = np.triu(np.ones((n, n), dtype=float), k=1)

    # Demo a single query position i.
    i = 6
    rng = np.random.default_rng(7)
    scores = rng.normal(loc=0.0, scale=1.0, size=n)

    attn_unmasked = softmax(scores)

    # Apply mask to scores: masked positions become -inf.
    masked_scores = scores.copy()
    masked_scores[mask[i] == 1.0] = -1e9
    attn_masked = softmax(masked_scores)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax0, ax1 = axes

    im = ax0.imshow(mask, cmap="Greys", aspect="auto")
    ax0.set_title("Causal mask M (upper-triangular)")
    ax0.set_xlabel("Key position j")
    ax0.set_ylabel("Query position i")
    ax0.set_xticks(range(0, n, 2))
    ax0.set_yticks(range(0, n, 2))

    # Highlight the demo row
    ax0.axhline(i, color="#6C8EBF", linewidth=2, alpha=0.9)

    ax1.set_title(f"Masked vs Unmasked attention (query i={i})")
    x = np.arange(n)
    width = 0.42
    ax1.bar(x - width / 2, attn_unmasked, width=width, label="unmasked", color="#DAE8FC", edgecolor="#6C8EBF")
    ax1.bar(x + width / 2, attn_masked, width=width, label="masked", color="#F8CECC", edgecolor="#B85450")
    ax1.set_xlabel("Key position j")
    ax1.set_ylabel("Attention weight")
    ax1.set_ylim(0, max(attn_unmasked.max(), attn_masked.max()) * 1.15)
    ax1.legend()

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "images",
        "causal_mask_demo.png",
    )
    out_path = os.path.abspath(out_path)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

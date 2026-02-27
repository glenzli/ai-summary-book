import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def kv_cache_bytes(seq_len: np.ndarray, n_layers: int, n_heads: int, head_dim: int, bytes_per_elem: int = 2) -> np.ndarray:
    """Approx KV cache memory for decoder-only attention.

    K/V per layer per token: n_heads * head_dim.
    Total: 2 (K+V) * layers * seq_len * n_heads * head_dim * bytes.
    """
    return 2 * n_layers * seq_len * n_heads * head_dim * bytes_per_elem


def main():
    seq = np.linspace(1_000, 128_000, 200)

    configs = [
        {"name": "32L, 32H, d_h=128 (≈4K)", "L": 32, "H": 32, "D": 128, "color": "#6C8EBF"},
        {"name": "48L, 40H, d_h=128 (≈5K)", "L": 48, "H": 40, "D": 128, "color": "#9673A6"},
        {"name": "80L, 64H, d_h=128 (≈8K)", "L": 80, "H": 64, "D": 128, "color": "#B85450"},
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    for cfg in configs:
        mem_bytes = kv_cache_bytes(seq, cfg["L"], cfg["H"], cfg["D"], bytes_per_elem=2)
        mem_gb = mem_bytes / (1024**3)
        ax.plot(seq, mem_gb, label=cfg["name"], color=cfg["color"], linewidth=2)

    ax.set_title("KV Cache Memory Growth (FP16, approximate)")
    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("KV cache memory (GB)")
    ax.legend(loc="upper left")

    # A few reference lines
    for ref_gb in [8, 16, 24, 48]:
        ax.axhline(ref_gb, color="#666666", linewidth=1, alpha=0.2)

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "images",
        "kv_cache_memory_curve.png",
    )
    out_path = os.path.abspath(out_path)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

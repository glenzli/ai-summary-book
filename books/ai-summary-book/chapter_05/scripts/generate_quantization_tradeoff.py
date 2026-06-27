import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    # Illustrative: weight-only memory for a 70B-parameter model.
    params = 70e9

    formats = ["FP16", "INT8", "INT4"]
    bits = np.array([16, 8, 4])
    bytes_per_param = bits / 8

    mem_gb = params * bytes_per_param / (1024**3)

    # Toy quality score: higher precision tends to retain more accuracy.
    quality = np.array([1.00, 0.985, 0.965])

    fig, ax1 = plt.subplots(figsize=(8.6, 4.6))

    x = np.arange(len(formats))
    bars = ax1.bar(x, mem_gb, color=["#6C8EBF", "#D6B656", "#B85450"], edgecolor="#666666")
    ax1.set_xticks(x)
    ax1.set_xticklabels(formats)
    ax1.set_ylabel("Weight memory (GB)")
    ax1.set_title("Quantization tradeoff (illustrative, weight-only)")

    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f} GB", ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, quality, marker="o", linewidth=2.5, color="#82B366")
    ax2.set_ylim(0.9, 1.02)
    ax2.set_ylabel("Quality (toy, normalized)")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "quantization_tradeoff.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    labels = ["[MASK] (80%)", "Random (10%)", "Keep (10%)"]
    values = [80, 10, 10]
    colors = ["#6C8EBF", "#D6B656", "#B85450"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="#666666", linewidth=1)

    ax.set_title("BERT MLM masking: 15% selected tokens, 80/10/10 strategy")
    ax.set_ylabel("Percent within selected tokens")
    ax.set_ylim(0, 100)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.25)

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "mlm_masking_strategy.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

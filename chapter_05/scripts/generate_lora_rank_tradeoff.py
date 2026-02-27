import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    # A toy tradeoff plot for LoRA rank r.
    # Assume we adapt N matrices of shape d x d.
    d = 4096
    n_mats = 4  # e.g., Wq, Wk, Wv, Wo per layer (illustrative)

    r = np.array([1, 2, 4, 8, 16, 32, 64, 128])

    full_params = n_mats * (d * d)
    lora_params = n_mats * (2 * d * r)  # BA where B:d×r, A:r×d
    ratio = lora_params / full_params * 100.0

    # Toy "quality" curve: diminishing returns with rank.
    quality = 1.0 - np.exp(-r / 24.0)

    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))

    ax1.plot(r, ratio, marker="o", linewidth=2.5, color="#6C8EBF", label="Trainable params (% of full)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("LoRA rank r (log2 scale)")
    ax1.set_ylabel("Trainable params (%)")

    ax2 = ax1.twinx()
    ax2.plot(r, quality, marker="s", linewidth=2.5, color="#82B366", label="Quality (toy, normalized)")
    ax2.set_ylabel("Quality (normalized)")
    ax2.set_ylim(0, 1.05)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")

    ax1.set_title("LoRA intuition: rank r vs trainable parameters and returns")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "lora_rank_tradeoff.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

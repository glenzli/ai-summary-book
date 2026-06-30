import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    # β controls how strongly we penalize KL divergence from the reference model.
    beta = np.logspace(-3, 0, 60)

    # Toy curves for intuition (not empirical):
    # - As β increases, KL is pushed down.
    # - Reward tends to decrease because optimization is constrained.
    kl = 3.2 / (1.0 + 6.0 * beta) + 0.05
    reward = 1.0 - 0.22 * np.log10(1.0 + 30 * beta)

    fig, ax1 = plt.subplots(figsize=(8.8, 4.6))

    ax1.set_xscale("log")
    ax1.plot(beta, reward, color="#82B366", linewidth=2.5, label="Reward (toy)")
    ax1.set_xlabel("KL penalty weight β (log scale)")
    ax1.set_ylabel("Reward (relative)")
    ax1.set_ylim(0.5, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(beta, kl, color="#B85450", linewidth=2.5, label="KL divergence (toy)")
    ax2.set_ylabel("KL divergence (relative)")
    ax2.set_ylim(0.0, max(kl) * 1.1)

    # Unified legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")

    ax1.set_title("RLHF intuition: β trades off reward vs divergence from reference")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "kl_anchor_tradeoff.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

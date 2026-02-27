import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style  # noqa: F401


def main():
    original = "The cute dog runs in the park ."
    masked_input = "The <X> runs in the <Y> ."
    target = "<X> cute dog <Y> park <Z>"

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")

    ax.text(0.02, 0.78, "Original:", fontsize=11, weight="bold")
    ax.text(0.18, 0.78, original, fontsize=11)

    ax.text(0.02, 0.50, "Input:", fontsize=11, weight="bold")
    ax.text(0.18, 0.50, masked_input, fontsize=11)

    ax.text(0.02, 0.22, "Target:", fontsize=11, weight="bold")
    ax.text(0.18, 0.22, target, fontsize=11)

    ax.set_title("T5 Span Corruption (Text Infilling) Example", pad=10)

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "span_corruption_example.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
This script reads the transfer learning results CSV
and saves before-vs-after bar graph.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_PATH = Path("outputs") / "transfer_learning" / "basic" / "results.csv"
PLOT_PATH = Path("outputs") / "transfer_learning" / "basic" / "before_after_bar_graph.png"


def load_results(results_path):
    rows = list(csv.DictReader(results_path.open("r", newline="", encoding="utf-8")))
    before_row = next(row for row in rows if "before" in row["stage"].lower())
    after_row = next(row for row in rows if "after" in row["stage"].lower())
    return before_row, after_row


def save_bar_graph(plot_path, before_row, after_row):
    labels = ["Before", "After"]
    values = [
        float(before_row["overall_accuracy_percent"]),
        float(after_row["overall_accuracy_percent"]),
    ]
    colors = ["#4C78A8", "#59A14F"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Before vs After Transfer Learning")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main():
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    before_row, after_row = load_results(RESULTS_PATH)
    save_bar_graph(PLOT_PATH, before_row, after_row)


if __name__ == "__main__":
    main()

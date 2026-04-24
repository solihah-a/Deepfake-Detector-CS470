import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


outputs_dir = "outputs"


def main():
    parser = argparse.ArgumentParser(
        description="show a simple bar chart from saved model results."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join("outputs", "d3_pretrained_results"),
        help="folder with the saved y_pred and y_true files.",
    )
    parser.add_argument(
        "--plot-path",
        default=os.path.join(outputs_dir, "simple_accuracy_comparison.png"),
        help="where to save the bar chart image.",
    )
    parser.add_argument(
        "--title",
        default="accuracy by group",
        help="simple chart title.",
    )
    args = parser.parse_args()

    os.makedirs(outputs_dir, exist_ok=True)

    results_dir = args.results_dir
    y_pred_path = os.path.join(results_dir, "None_None_ypred.npz")
    y_true_path = os.path.join(results_dir, "None_None_ytrue.npz")
    plot_path = args.plot_path

    y_pred = np.load(y_pred_path)["arr_0"]
    y_true = np.load(y_true_path)["arr_0"]

    real_acc = float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5))
    fake_acc = float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5))
    overall_acc = float(accuracy_score(y_true, y_pred > 0.5))

    print("total images:", len(y_true))
    print("real images :", int((y_true == 0).sum()))
    print("fake images :", int((y_true == 1).sum()))
    print()
    print("--- threshold = 0.5 ---")
    print("real acc:", round(real_acc * 100, 2), "%")
    print("fake acc:", round(fake_acc * 100, 2), "%")
    print("overall :", round(overall_acc * 100, 2), "%")

    labels = ["real", "fake", "overall"]
    values = [float(real_acc * 100), float(fake_acc * 100), float(overall_acc * 100)]
    colors = ["#4C78A8", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value + 1),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(args.title.lower())
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print()
    print("saved plot:", os.path.abspath(plot_path))


if __name__ == "__main__":
    main()

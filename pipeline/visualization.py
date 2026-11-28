"""Plot learning curves and classification results."""

# Training and validation loss by epoch
# Gradient norm by training step with updates during training
# Test loss by batch and mean test loss
# Confusion matrix and per-class results

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training(run_dir):
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in run_dir.glob("epoch-*.json")
    ]
    if not records:
        raise SystemExit(f"No epoch files found in {run_dir}")

    records.sort(key=lambda record: record["epoch"])
    epochs = [record["epoch"] for record in records]
    train_loss = [record["train_loss"] for record in records]
    val_loss = [record["validation_loss"] for record in records]
    accuracy = [100 * record["validation_accuracy"] for record in records]
    grad_norms = [
        batch["grad_norm"]
        for record in records
        for batch in record["batches"]
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), layout="constrained")

    axes[0].plot(epochs, train_loss, marker="o", label="Training")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Mean cross-entropy loss", xticks=epochs)
    axes[0].legend()

    axes[1].plot(epochs, accuracy, marker="o", color="tab:green")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)",
                xticks=epochs, ylim=(0, 100))

    steps = range(1, len(grad_norms) + 1)
    axes[2].plot(steps, grad_norms, linewidth=0.7)
    axes[2].set(xlabel="Training batch", ylabel="Global L2 gradient norm")

    output = run_dir / "training_curves.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


def plot_confusion_matrix(run_dir):
    import numpy as np

    results = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    counts = np.array(results["confusion_matrix"])
    class_ids = results["class_ids"]

    # Each row shows how the model classified one actual sign class.
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = 100 * counts / np.maximum(row_totals, 1)

    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    heatmap = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    positions = range(len(class_ids))
    ax.set_xticks(positions, labels=class_ids, rotation=90, fontsize=7)
    ax.set_yticks(positions, labels=class_ids, fontsize=7)
    ax.set_xlabel("Predicted sign class")
    ax.set_ylabel("Actual sign class")
    ax.set_title("Traffic sign classification on the official test set")
    fig.colorbar(heatmap, ax=ax, label="Percent of actual class")

    output = run_dir / "confusion_matrix.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")

    errors = counts.copy()
    np.fill_diagonal(errors, 0)
    print("Most frequent mistakes by image count:")
    for index in np.argsort(errors.ravel())[::-1][:5]:
        actual, predicted = np.unravel_index(index, errors.shape)
        count = errors[actual, predicted]
        if count == 0:
            break
        print(f"Actual {class_ids[actual]} predicted as {class_ids[predicted]}: "
              f"{count} images ({percentages[actual, predicted]:.1f}% of that class)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a saved classification run.")
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir
    plot_training(run_dir)
    if (run_dir / "test_metrics.json").is_file():
        plot_confusion_matrix(run_dir)

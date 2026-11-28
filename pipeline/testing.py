"""Evaluate the selected checkpoint on the held-out test split."""

# Test loss per batch and across the full split
# Accuracy, macro F1, and per-class results

import argparse
from itertools import batched
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from tqdm import tqdm

from pipeline.data_ingestion import DATASET_DIR
from pipeline.data_normalization import iter_normalized_pairs
from pipeline.data_preparation import prepare_dataframes
from pipeline.model import build_model


@torch.inference_mode()
def evaluate_test(checkpoint_path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    _, _, test = prepare_dataframes(DATASET_DIR, seed=saved["seed"])
    pairs = iter_normalized_pairs(test, saved["mean"], saved["std"])
    progress = tqdm(pairs, total=len(test), desc="Testing", unit="img")

    actual = []
    predicted = []
    total_loss = 0.0
    batch_losses = []

    for batch in batched(progress, saved["batch_size"]):
        images, labels = zip(*batch)
        scores = model(torch.stack(images))
        targets = torch.tensor(labels, dtype=torch.long)

        loss_sum = nn.functional.cross_entropy(scores, targets, reduction="sum").item()
        total_loss += loss_sum
        batch_losses.append(loss_sum / len(batch))
        actual.extend(labels)
        predicted.extend(scores.argmax(dim=1).tolist())

    class_ids = list(range(43))
    matrix = confusion_matrix(actual, predicted, labels=class_ids)
    report = classification_report(
        actual, predicted, labels=class_ids, output_dict=True, zero_division=0
    )

    correct = int(matrix.trace())
    count = len(actual)
    results = {
        "checkpoint_epoch": saved["epoch"],
        "test_images": count,
        "correct": correct,
        "mean_loss": total_loss / count,
        "accuracy": correct / count,
        "macro_f1": report["macro avg"]["f1-score"],
        "batch_losses": batch_losses,
        "class_ids": class_ids,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    output = checkpoint_path.parent / "test_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Correct predictions: {correct:,} / {count:,}")
    print(f"Mean test loss:      {results['mean_loss']:.4f}")
    print(f"Test accuracy:       {results['accuracy']:.2%}")
    print(f"Macro F1:            {results['macro_f1']:.4f}")
    print(f"Saved:               {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the selected checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    evaluate_test(parser.parse_args().checkpoint)

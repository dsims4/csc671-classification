"""Train and validate the traffic sign classifier."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import torch

from environment import get_device
from pipeline.data_ingestion import DATASET_DIR
from pipeline.data_normalization import fit_normalization
from pipeline.data_preparation import prepare_dataframes
from pipeline.model import build_model
from pipeline.training import train_one_epoch
from pipeline.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    seed = 42
    batch_size = 64
    learning_rate = 0.01
    torch.manual_seed(seed)
    device = get_device()

    print(f"Preparing data on {device}...", flush=True)
    train, validation, _ = prepare_dataframes(DATASET_DIR, seed=seed)
    print(f"Calculating RGB statistics from {len(train):,} training signs...", flush=True)
    mean, std = fit_normalization(train)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    run_dir = Path("artifacts") / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        started = perf_counter()
        train_loss, batches = train_one_epoch(
            model, optimizer, train, mean, std,
            batch_size=batch_size, seed=seed + epoch - 1,
        )

        print(f"Scoring {len(validation):,} validation signs...", flush=True)
        val_loss, val_accuracy = validate(
            model, validation, mean, std, batch_size=batch_size,
        )
        elapsed = perf_counter() - started

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "epoch_seconds": elapsed,
            "batches": batches,
        }
        (run_dir / f"epoch-{epoch}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8",
        )

        print(f"Training loss:       {train_loss:.4f}")
        print(f"Validation loss:     {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Epoch time:          {elapsed:.1f} seconds")

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = run_dir / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_loss": val_loss,
                "mean": mean,
                "std": std,
                "seed": seed,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }, checkpoint)
            print(f"Saved checkpoint:    {checkpoint}", flush=True)


if __name__ == "__main__":
    main()

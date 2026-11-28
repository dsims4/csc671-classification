"""Measure classifier loss and accuracy on the validation split."""

from itertools import batched

import torch
from torch import nn

from pipeline.data_normalization import iter_normalized_pairs


@torch.inference_mode()
def validate(model, frame, mean, std, *, batch_size=64):
    """Return mean loss per image and accuracy as a fraction from 0 to 1."""
    if frame.empty:
        raise ValueError("Validation requires at least one image.")

    model.eval()
    device = next(model.parameters()).device
    pairs = iter_normalized_pairs(frame, mean, std)
    total_loss = 0.0
    total_correct = 0

    for batch in batched(pairs, batch_size):
        images, labels = zip(*batch)
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        scores = model(images)
        total_loss += nn.functional.cross_entropy(
            scores, labels, reduction="sum"
        ).item()
        total_correct += (scores.argmax(dim=1) == labels).sum().item()

    return total_loss / len(frame), total_correct / len(frame)

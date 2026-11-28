"""Train the traffic sign classifier and record batch metrics."""

from itertools import batched

import torch
from torch import nn
from tqdm import tqdm

from pipeline.data_normalization import iter_normalized_pairs


def train_one_epoch(model, optimizer, frame, mean, std, *, batch_size=64, seed=42):
    """Train on each row once and return mean loss per image and batch metrics."""
    if frame.empty:
        raise ValueError("Training requires at least one image.")

    model.train()
    device = next(model.parameters()).device
    shuffled = frame.sample(frac=1, random_state=seed)
    pairs = iter_normalized_pairs(shuffled, mean, std)
    total_loss = 0.0
    history = []

    with tqdm(total=len(frame), desc="Training", unit="img") as progress:
        for batch in batched(pairs, batch_size):
            images, labels = zip(*batch)
            images = torch.stack(images).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = nn.functional.cross_entropy(scores, labels)
            loss.backward()

            grad_norms = [
                parameter.grad.detach().norm()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            grad_norm = torch.stack(grad_norms).norm().item()
            optimizer.step()

            batch_loss = loss.item()
            history.append({"loss": batch_loss, "grad_norm": grad_norm})
            total_loss += batch_loss * len(batch)
            progress.update(len(batch))

    return total_loss / len(frame), history

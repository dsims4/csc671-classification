"""Define a compact classifier for 32 x 32 RGB traffic sign images."""

from torch import nn


def build_model() -> nn.Sequential:
    """Return one raw score per traffic sign class."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 43),
    )

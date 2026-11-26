"""Normalize RGB inputs using training-set statistics."""

from collections.abc import Iterator

import pandas as pd
import torch

from pipeline.data_ingestion import DATASET_DIR
from pipeline.data_preparation import iter_image_pairs, prepare_dataframes


def fit_normalization(train_frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit channel statistics in one pass over the training images.

    Args:
        train_frame: Prepared training rows with image_path and class_id columns.
            Pass only the training split to keep evaluation data held out.

    Returns:
        A pair of length-three float32 tensors holding the mean and population
        standard deviation in RGB order on the [0, 1] pixel scale.

    Raises:
        OSError: If a training image cannot be opened or decoded.
        ValueError: If the training frame is empty or a channel has zero variance.
    """
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros_like(channel_sum)
    pixel_count = 0

    for image, _ in iter_image_pairs(train_frame):
        # Sum byte values before scaling so constant channels have exactly zero variance.
        pixels = image.flatten(1).to(torch.float64)
        channel_sum += pixels.sum(dim=1)
        channel_squared_sum += pixels.square().sum(dim=1)
        pixel_count += pixels.shape[1]

    if pixel_count == 0:
        raise ValueError("Normalization requires at least one training image.")

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean.square()
    std = variance.clamp_min(0).sqrt()
    if (std == 0).any():
        raise ValueError("Each RGB channel must have nonzero variance in the training split.")
    return (mean / 255).float(), (std / 255).float()


def iter_normalized_pairs(
    frame: pd.DataFrame, mean: torch.Tensor, std: torch.Tensor
) -> Iterator[tuple[torch.Tensor, int]]:
    """Scale and standardize each image on demand.

    Args:
        frame: Prepared training, validation, or test rows with image_path
            and class_id columns.
        mean: Length-three float32 tensor of fitted training means in RGB order.
        std: Length-three float32 tensor of positive training standard deviations
            in RGB order.

    Yields:
        Each standardized float32 tensor of shape (3, 32, 32) with its class ID.

    Raises:
        OSError: If an image cannot be opened or decoded during iteration.
    """
    # Singleton height and width apply one mean and scale across each channel.
    mean = mean[:, None, None]
    std = std[:, None, None]
    for image, class_id in iter_image_pairs(frame):
        pixels = image.to(torch.float32).div_(255.0)
        yield (pixels - mean) / std, class_id


def main() -> None:
    """Fit statistics on the default training split and print them in RGB order.

    Data preparation and normalization errors propagate to the command line.
    """
    train_frame, _, _ = prepare_dataframes(DATASET_DIR)
    mean, std = fit_normalization(train_frame)
    print(f"RGB mean {mean.tolist()}")
    print(f"RGB standard deviation {std.tolist()}")


if __name__ == "__main__":
    main()

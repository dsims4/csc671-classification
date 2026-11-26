"""Prepare validated images and labels for classification."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

from pipeline.data_ingestion import DATASET_DIR
from pipeline.data_validation import ImageLabelPair, validate_dataset


IMAGE_SIZE = (32, 32)


def prepare_dataframe(pairs: list[ImageLabelPair]) -> pd.DataFrame:
    """Build a DataFrame index from validated image and label pairs.

    Args:
        pairs: PNG paths and their integer class IDs in the desired row order.

    Returns:
        A DataFrame with Path objects in image_path and int64 labels in class_id.
    """
    frame = pd.DataFrame(pairs, columns=("image_path", "class_id"))
    return frame.astype({"class_id": "int64"})


def prepare_dataframes(dataset_dir: Path, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare the training, validation, and official test DataFrames.

    The first of five stratified group folds is used for validation so that
    different views of one physical sign stay together.

    Args:
        dataset_dir: Root of the extracted GTSRB dataset.
        seed: Random seed for reproducible fold assignment.

    Returns:
        The training, validation, and test DataFrames in that order. Each
        preserves its original CSV row indices and contains paths and labels.

    Raises:
        OSError: If a dataset manifest cannot be read.
        csv.Error: If a dataset manifest cannot be parsed.
        ValueError: If a manifest fails validation or the split cannot be formed.
    """
    train_pairs, test_pairs = validate_dataset(dataset_dir)
    training_pool = prepare_dataframe(train_pairs)
    # Track numbers restart in each class. The class and track together identify a sign.
    track_ids = training_pool["image_path"].map(lambda path: path.stem.rsplit("_", 1)[0])

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, validation_indices = next(splitter.split(training_pool, training_pool["class_id"], track_ids))
    return training_pool.iloc[train_indices], training_pool.iloc[validation_indices], prepare_dataframe(test_pairs)


def load_image(image_path: Path) -> Tensor:
    """Decode one image as RGB and resize it with bilinear interpolation.

    Args:
        image_path: Local path to the source PNG.

    Returns:
        A CPU uint8 tensor of shape (3, 32, 32) in RGB channel order with
        values from 0 to 255.

    Raises:
        OSError: If the image cannot be opened or decoded by Pillow.
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return pil_to_tensor(image)


def iter_image_pairs(frame: pd.DataFrame) -> Iterator[tuple[Tensor, int]]:
    """Load images on demand in the DataFrame's row order.

    Args:
        frame: A prepared DataFrame containing image_path and class_id columns.

    Yields:
        Each resized RGB byte tensor paired with its integer class ID.

    Raises:
        OSError: If an image cannot be opened or decoded during iteration.
    """
    for row in frame.itertuples(index=False):
        yield load_image(row.image_path), row.class_id


def main() -> None:
    """Prepare the default dataset splits and print their row counts.

    Dataset validation and split errors from prepare_dataframes propagate to
    the command line.
    """
    train_frame, validation_frame, test_frame = prepare_dataframes(DATASET_DIR)
    print(
        f"Prepared {len(train_frame):,} training rows, {len(validation_frame):,} validation rows, "
        f"and {len(test_frame):,} test rows."
    )


if __name__ == "__main__":
    main()

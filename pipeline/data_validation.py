"""Validate the image and label pairs in the GTSRB train and test splits."""

import csv
from pathlib import Path
import re

from pipeline.data_ingestion import DATASET_DIR


TRAIN_FILENAME = re.compile(r"([0-9]{5})_[0-9]{5}_[0-9]{5}")

type ImageLabelPair = tuple[Path, int]


def _read_pairs(dataset_dir: Path, split: str, expected_count: int) -> list[ImageLabelPair]:
    """Validate image and label entries in one official CSV manifest.

    Args:
        dataset_dir: Directory containing the manifests and image folders.
        split: Official split name, either Train or Test.
        expected_count: Number of distinct image paths required in the split.

    Returns:
        Pairs of local PNG paths and integer class IDs in CSV row order.

    Raises:
        OSError: If the manifest cannot be read.
        csv.Error: If the CSV reader cannot parse the manifest.
        ValueError: If a required column, label, image path, training filename,
            or row count fails validation.
    """
    csv_path = dataset_dir / f"{split}.csv"
    pairs: dict[Path, int] = {}

    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        missing_columns = {"Path", "ClassId"} - set(rows.fieldnames or ())
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path}: missing columns {names}")

        for row in rows:
            location = f"{csv_path}:{rows.line_num}"
            relative_path = Path(row["Path"] or "")
            try:
                class_id = int(row["ClassId"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{location}: invalid ClassId {row['ClassId']!r} for {relative_path}") from error
            if class_id not in range(43):
                raise ValueError(f"{location}: ClassId {class_id} is outside 0–42 for {relative_path}")

            expected_parent = Path("Train") / str(class_id) if split == "Train" else Path("Test")
            if relative_path.parent != expected_parent or relative_path.suffix.lower() != ".png":
                raise ValueError(f"{location}: expected a PNG path under {expected_parent}, got {relative_path}")
            if split == "Train":
                match = TRAIN_FILENAME.fullmatch(relative_path.stem)
                if match is None or int(match[1]) != class_id:
                    raise ValueError(
                        f"{location}: expected a class_track_frame filename for class {class_id}, got {relative_path}"
                    )

            image_path = dataset_dir / relative_path
            if image_path in pairs:
                raise ValueError(f"{location}: duplicate image path {relative_path}")
            if not image_path.is_file():
                raise ValueError(f"{location}: missing image {relative_path}")
            pairs[image_path] = class_id

    if len(pairs) != expected_count:
        raise ValueError(f"{csv_path}: expected {expected_count:,} pairs, found {len(pairs):,}")
    return list(pairs.items())


def validate_dataset(dataset_dir: Path) -> tuple[list[ImageLabelPair], list[ImageLabelPair]]:
    """Validate the official training and test manifests.

    Args:
        dataset_dir: Root of the extracted GTSRB dataset.

    Returns:
        The 39,209 training pairs followed by the 12,630 test pairs as two
        lists. Each item contains a local PNG path and an integer class ID.

    Raises:
        OSError: If either manifest cannot be read.
        csv.Error: If either manifest cannot be parsed.
        ValueError: If the labels, image paths, or manifest structure are invalid.
    """
    train_pairs = _read_pairs(dataset_dir, "Train", 39_209)
    test_pairs = _read_pairs(dataset_dir, "Test", 12_630)
    return train_pairs, test_pairs


def main() -> None:
    """Validate the local dataset and print the training and test pair counts.

    Manifest and image-path errors from validate_dataset propagate to the
    command line.
    """
    train_pairs, test_pairs = validate_dataset(DATASET_DIR)
    print(f"Validated {len(train_pairs):,} training pairs and {len(test_pairs):,} test pairs.")


if __name__ == "__main__":
    main()

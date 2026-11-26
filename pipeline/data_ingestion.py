"""Fetch the GTSRB dataset from Kaggle."""

from pathlib import Path

import kagglehub


DATASET_HANDLE = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "gtsrb"


def download_dataset() -> Path:
    """Fetch and extract GTSRB with KaggleHub when needed.

    Returns:
        The local dataset directory containing the images and CSV manifests.

    Raises:
        kagglehub.exceptions.KaggleApiHTTPError: If Kaggle rejects the request.
        kagglehub.exceptions.DataCorruptionError: If the archive fails its
            download checksum.
        OSError: If the local files cannot be written.
    """
    return Path(kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(DATASET_DIR)))


def main() -> None:
    """Download or reuse the dataset and print its local directory.

    Download and extraction errors from download_dataset propagate to the
    command line.
    """
    print(download_dataset())


if __name__ == "__main__":
    main()

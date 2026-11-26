# csc671-classification

Train a compact Conv2d classifier from scratch on Kaggle's German Traffic Sign Recognition Benchmark (GTSRB), aiming for high accuracy with low compute, memory use, and overfitting.

## Environment setup

Use Python **3.14.10**, pinned in `.python-version`, and a separate local virtual environment on each machine. Choose the PyTorch build for your hardware.

OpenAI Codex assisted with environment setup, package installation, version documentation, and hardware compatibility checks to speed up training and testing workflows. This acknowledgment covers environment setup only.

On macOS or Linux:

```bash
git clone https://github.com/dsims4/csc671-classification.git
cd csc671-classification
python3.14 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `py -3.14 -m venv .venv` and `.venv\Scripts\Activate.ps1`. Verify `python --version` is 3.14.10. If using pyenv, install the pinned version first. `.venv` is local and ignored by Git.

Install **one** PyTorch backend below before installing the remaining requirements. Start with a fresh virtual environment when changing backends.

| Hardware | Platform | Install command |
| --- | --- | --- |
| Apple Silicon, MPS or CPU | macOS 14+ | `python -m pip install -r requirements-torch.txt` |
| CPU | Supported Linux or Windows | `python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA GPU, CUDA 13.2 | Supported Linux or Windows, compatible GPU and driver | `python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu132` |
| AMD GPU, ROCm 7.14 | Supported Linux x86_64, compatible GPU and driver | `python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/rocm7.14` |

These are available wheel combinations, not a guarantee that every GPU or operating system is supported. Check the [PyTorch installation guidance](https://pytorch.org/get-started/locally/) and [AMD compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) against the actual machine. CPU is the fallback on supported systems. Native Windows ROCm and Intel macOS are not covered by these pinned installation commands.

Then install the shared packages and check the environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python environment.py
```

`requirements.txt` includes the framework pins from `requirements-torch.txt`. Installing the backend first preserves its matching `+cpu`, `+cu132`, or `+rocm7.14` build. Do not copy `.venv` between machines.

## Package versions

Direct dependencies are pinned below and in the requirements files. Transitive dependencies are resolved by pip; save a full freeze with each experiment for an exact environment record.

| Package | Version | Purpose |
| --- | --- | --- |
| torch | 2.15.0 | Conv2d layers, automatic gradients, optimizers, device support |
| torchvision | 0.29.2 | Image conversion and augmentation; custom dataset loading |
| numpy | 2.5.5 | Array operations and dataset statistics |
| pandas | 3.1.1 | Image paths and class labels in DataFrames |
| pillow | 13.0.0 | Image decoding and RGB conversion |
| scikit-learn | 1.9.1 | Grouped validation split, class metrics, confusion matrix |
| matplotlib | 3.11.2 | Learning curves and error plots |
| kagglehub | 1.0.3 | Download the Kaggle dataset |
| tqdm | 4.70.1 | Training and loading progress |

Use Python's built-in `csv`, `json`, and `pathlib` for metadata, settings, and paths. Experiment records can use CSV/JSON without a separate tracking service.

## Hardware and compatibility

Record each teammate's hardware with `environment.py`. It reports hardware, runtimes, and package versions and checks a synthetic Conv2d training step.

```bash
mkdir -p artifacts
python environment.py > artifacts/environment.json
python -m pip freeze > artifacts/packages.txt
python environment.py --device cpu
```

On Windows, create `artifacts` with `New-Item -ItemType Directory -Force artifacts`. Keep NVIDIA or AMD driver versions alongside the report, using the machine's `nvidia-smi` or `amd-smi` output when available. A CPU wheel cannot verify an installed GPU's accelerator support.

`get_device()` in `environment.py` selects CUDA/ROCm, then MPS, then CPU; explicit `--device` requests fail clearly when unavailable. ROCm uses PyTorch's `cuda` device API and is identified by `torch.version.hip`, as described in the [HIP semantics](https://docs.pytorch.org/docs/2.15/notes/hip.html).

## Dataset

Use [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign): 43 classes, with an expected 39,209 training images and 12,630 test images. Verify these counts, labels, and paths after download. This is classification of cropped signs.

Run the data stages from the repository root. The ingestion module uses KaggleHub to download the files to `data/gtsrb/`.

```bash
python -m pipeline.data_ingestion
python -m pipeline.data_validation
python -m pipeline.data_preparation
python -m pipeline.data_normalization
```

`Train.csv` and `Test.csv` supply the labels and relative paths for `Train/` and `Test/`. `Meta/` and `Meta.csv` supply class references. Record the dataset version and file hashes with the experiment. If Kaggle requires authentication, use `kagglehub.login()` locally. Keep credentials outside Git.

Each training filename identifies its class, track, and frame. Validation checks that its class matches the CSV label and directory. Preparation keeps each `(class, track)` group in one fold so validation measures performance on physical signs held out from training. It uses one of five folds for validation, with seed `42` and approximate class stratification. The official test set is reserved for final evaluation. [Original benchmark documentation](https://github.com/houbensebastian/GermanTrafficSignBenchmarks)

The DataFrames hold paths and integer class labels. The loader decodes images on demand and uses bilinear resizing to produce uniform 32 × 32 RGB inputs for batching. Normalization divides pixel values by `255.0`, then subtracts the mean and divides by the population standard deviation for each channel. The statistics come from the training fold and are reused for validation and testing.

## Experiment methodology

1. Train a compact Conv2d classifier from scratch. Record training loss and the global L2 gradient norm after backpropagation. Measure validation loss and accuracy after each epoch.
2. Select the checkpoint with the lowest validation loss. Evaluate it on the official test split and report mean loss per image, accuracy, macro F1, and a confusion matrix.
3. Compare runs using the same split. Record epoch time, peak memory, and run settings alongside the results.

Record experiment results under `artifacts/`. Dataset files and generated training outputs are ignored by Git. `main.py` is currently a starter entry point.

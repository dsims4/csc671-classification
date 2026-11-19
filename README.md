# csc671-classification

Train a compact Conv2d classifier from scratch on Kaggle's German Traffic Sign Recognition Benchmark (GTSRB), aiming for high accuracy with low compute, memory use, and overfitting. The architecture and settings below are proposals; no accuracy results have been measured.

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
| pillow | 13.0.0 | Image decoding and RGB conversion |
| scikit-learn | 1.9.1 | Grouped validation split, class metrics, confusion matrix |
| matplotlib | 3.11.2 | Learning curves and error plots |
| kagglehub | 1.0.3 | Download the Kaggle dataset |
| tqdm | 4.70.1 | Training and loading progress |

Use Python's built-in `csv`, `json`, and `pathlib` for metadata, settings, and paths. Experiment records can use CSV/JSON without a separate tracking service.

## Hardware and compatibility

Verified development machine on 2026-11-19:

| Item | Observed value |
| --- | --- |
| Operating system | macOS 26.6.2 |
| Architecture | arm64 |
| CPU | Apple M1, 8 logical CPUs |
| System memory | 8 GiB, shared with the integrated GPU |
| GPU | Apple M1 integrated GPU |
| Python | 3.14.10 |
| PyTorch / torchvision | 2.15.0 / 0.29.2 |
| Available accelerator | MPS |
| CUDA / ROCm runtime | Not present in the macOS build |
| Verified operation | Conv2d forward pass, loss, backward pass, and optimizer step on MPS and CPU |

Record each teammate's hardware with `environment.py`. It reports hardware, runtimes, and package versions and checks a synthetic Conv2d training step. Only the machine above has been tested; CUDA and ROCm instructions have not been runtime-tested on team hardware.

```bash
mkdir -p artifacts
python environment.py > artifacts/environment.json
python -m pip freeze > artifacts/packages.txt
python environment.py --device cpu
```

On Windows, create `artifacts` with `New-Item -ItemType Directory -Force artifacts`. Keep NVIDIA or AMD driver versions alongside the report, using the machine's `nvidia-smi` or `amd-smi` output when available. A CPU wheel cannot verify an installed GPU's accelerator support.

`get_device()` in `environment.py` selects CUDA/ROCm, then MPS, then CPU; explicit `--device` requests fail clearly when unavailable. ROCm uses PyTorch's `cuda` device API and is identified by `torch.version.hip`, as described in the [HIP semantics](https://docs.pytorch.org/docs/2.15/notes/hip.html).

For the training implementation, move models and batches to the selected device, keep checkpoint tensors portable, and use a guarded `main()` for Windows data loaders. Start with FP32 and `num_workers=0` everywhere. Make workers, batch size, pinning, and mixed precision configurable after profiling. Save model weights with settings and normalization statistics; load with `map_location` on another device. The same seed does not guarantee identical floating-point results across hardware.

## Dataset

Use [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign): 43 classes, with an expected 39,209 training images and 12,630 test images. Verify these counts, labels, and paths after download. This is classification of cropped signs.

Download when ready with the installed [KaggleHub client](https://github.com/Kaggle/kagglehub):

```python
import kagglehub

path = kagglehub.dataset_download(
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
    output_dir="data/gtsrb",
)
print(path)
```

The expected layout contains `Train/`, `Test/`, `Meta/`, `Train.csv`, and `Test.csv`. Use CSV labels and relative paths; do not treat `Meta/` reference images as training samples. Record the downloaded dataset version and file hashes with the experiment. If Kaggle requires authentication, use `kagglehub.login()` locally. Keep credentials outside Git.

The [original benchmark documentation](https://github.com/houbensebastian/GermanTrafficSignBenchmarks) describes multiple frames per physical sign. Preserve complete tracks in one split. Verify how the Kaggle filenames encode class, track, and frame before constructing group IDs.

## Basic experiment outline

1. **Audit and split.** Check image integrity, class counts, duplicates, dimensions, and track IDs. Make a fixed approximately 80/20 split of the official training partition with `(class_id, track_id)` groups intact. Start with one fold from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Verify all 43 classes are represented and no groups or duplicates cross splits; save the manifest. Reserve the official test set until the recipe is frozen.
2. **Prepare images.** Resize RGB images to 32 × 32 and normalize using training-fold statistics. Apply modest training-only rotation (±10°), translation (up to 10%), scale, brightness, and contrast changes. Keep evaluation deterministic; avoid flips or transformations that change sign meaning. Compare 48 × 48 only if its accuracy gain justifies the cost.
3. **Build a baseline.** Use three blocks, each with two padded 3 × 3 Conv2d layers (no bias), BatchNorm/ReLU after each convolution, and 2 × 2 max pooling. Widths 32 → 64 → 128, global average pooling, dropout 0.2, and a linear layer with 43 logits give **293,003 parameters**. Use random initialization and cross-entropy on logits.
4. **Train and monitor.** Start with AdamW, learning rate 0.001, weight decay 0.0001, effective batch size 128, and at most 60 epochs. Reduce the learning rate when validation loss stalls; early-stop after 8 unimproved epochs and restore the best validation-loss checkpoint. Log loss and accuracy. Assess overfitting with unaugmented training metrics in evaluation mode.
5. **Compare candidates.** Change one factor at a time: width, convolution count, resolution, augmentation, weight decay, or dropout. Increase capacity for underfitting; reduce it or strengthen regularization for overfitting. Keep the split, evaluation, and training budget fixed. Run finalists with three fixed seeds.
6. **Choose and report.** Report accuracy, macro F1, per-class recall, confusion matrix, learning curves, and variation across seeds. Measure parameters, checkpoint size, training time, peak memory, and inference speed. Provisionally prefer the smallest/fastest model within 0.2 percentage points of the best validation accuracy with comparable macro F1, accounting for seed variation. Evaluate the frozen choice on the official test set.

Benchmark on the same device with fixed precision, batch size, workers, warm-up, and timed scope; synchronize accelerators before reading timings. Report batch-1 latency separately from throughput. Cross-machine times are not architecture comparisons. Smaller microbatches with gradient accumulation save memory but change BatchNorm statistics. Try [mixed precision](https://docs.pytorch.org/docs/2.15/amp.html) after the FP32 baseline, measuring accuracy and speed.

Save each run's configuration, seed, split hash, dataset version, hardware report, package freeze, curves, metrics, and best checkpoint under `artifacts/`. Dataset files and generated training outputs are ignored by Git. `main.py` is currently a starter entry point; the full data pipeline and training loop are still to be implemented.

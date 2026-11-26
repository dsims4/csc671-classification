"""Report the local runtime and verify that PyTorch can train on it."""

import argparse
import ctypes
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess

import torch


def get_device(requested: str = "auto") -> torch.device:
    """Select an available PyTorch device for the hardware check.

    Args:
        requested: One of auto, cpu, cuda, or mps. Auto prefers CUDA
            (including ROCm), then MPS, then CPU.

    Returns:
        The selected torch.device.

    Raises:
        ValueError: If the requested device name is unknown or unavailable.
    """
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available or not available[requested]:
        raise ValueError(f"Device {requested!r} is unavailable on this machine.")
    return torch.device(requested)


def get_ram_bytes() -> int | None:
    """Read the machine's physical memory capacity.

    Returns:
        Total installed RAM in bytes, or None if the system query fails.
    """
    if platform.system() == "Windows":
        total = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(total)):
            return total.value * 1024
        return None
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def get_cpu_name() -> str:
    """Read the processor name using the current platform's hardware report.

    Returns:
        The CPU model when available, with the platform processor name or
        machine architecture as a fallback.
    """
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return result.stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def get_gpu_details() -> list[dict]:
    """Discover the GPUs visible to the local hardware check.

    Returns:
        A list with each GPU's name and, for CUDA or ROCm, memory capacity
        in GiB. An empty list means the available queries found no GPUs.
    """
    if torch.cuda.is_available():
        return [
            {
                "name": torch.cuda.get_device_name(index),
                "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return [
                {"name": item.get("sppci_model", item.get("_name")), "cores": item.get("spdisplays_cores")}
                for item in json.loads(result.stdout).get("SPDisplaysDataType", [])
            ]
        except (OSError, subprocess.SubprocessError, ValueError):
            return []
    return []


def main() -> None:
    """Run one synthetic Conv2d training step and print the environment report.

    The report is written to standard output as JSON. The optional --device
    command-line argument selects the hardware used for the training check.

    Raises:
        SystemExit: If command-line parsing fails or the device is unavailable.
        RuntimeError: If the synthetic training step fails in PyTorch.
        importlib.metadata.PackageNotFoundError: If a reported package is missing.
    """
    parser = argparse.ArgumentParser(description="Report hardware and package versions; check a Conv2d training step.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    try:
        device = get_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 43),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    outputs = model(torch.randn(2, 3, 32, 32, device=device))
    loss = torch.nn.functional.cross_entropy(outputs, torch.tensor([0, 1], device=device))
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    ram = get_ram_bytes()
    packages = ("torch", "torchvision", "numpy", "pillow", "scikit-learn", "matplotlib", "kagglehub", "tqdm")
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu": get_cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "ram_gib": round(ram / 2**30, 2) if ram is not None else None,
        "gpus": get_gpu_details(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "rocm_runtime": torch.version.hip,
        "mps_available": torch.backends.mps.is_available(),
        "packages": {name: version(name) for name in packages},
        "conv2d_training_step": "passed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

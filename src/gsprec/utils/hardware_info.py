"""
Hardware profiling utilities: collect CPU, GPU, and RAM information
in a cross-platform (Windows / Linux / macOS) manner.

Uses psutil for memory; torch for GPU; platform for CPU info.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Dict, Optional

import numpy as np

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# CPU / RAM helpers
# ---------------------------------------------------------------------------

def get_cpu_info() -> Dict:
    """Return CPU model, core count, and RAM."""
    info: Dict = {
        "cpu_model": platform.processor() or "unknown",
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "physical_cores": os.cpu_count(),
    }
    if _PSUTIL:
        try:
            info["physical_cores"] = psutil.cpu_count(logical=False) or os.cpu_count()
            info["logical_cores"] = psutil.cpu_count(logical=True)
            vm = psutil.virtual_memory()
            info["total_ram_GB"] = round(vm.total / (1024 ** 3), 2)
            info["available_ram_GB"] = round(vm.available / (1024 ** 3), 2)
        except Exception:
            pass
    return info


def rss_mb() -> float:
    """Current process RSS memory in MB (cross-platform)."""
    if _PSUTIL:
        try:
            proc = psutil.Process(os.getpid())
            return float(proc.memory_info().rss) / (1024 ** 2)
        except Exception:
            pass
    # Fallback: Linux /proc/self/status
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def system_ram_used_mb() -> float:
    """Total system RAM in use (MB)."""
    if _PSUTIL:
        try:
            vm = psutil.virtual_memory()
            return float(vm.used) / (1024 ** 2)
        except Exception:
            pass
    return rss_mb()


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def get_gpu_info() -> Dict:
    """Return GPU model and VRAM if available."""
    info: Dict = {"gpu_available": False}
    if not _TORCH:
        return info
    if not torch.cuda.is_available():
        return info

    info["gpu_available"] = True
    try:
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_vram_GB"] = round(props.total_memory / (1024 ** 3), 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    except Exception:
        pass
    return info


def gpu_memory_allocated_mb() -> float:
    """GPU memory currently allocated by PyTorch (MB)."""
    if not _TORCH or not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.memory_allocated()) / (1024 ** 2)
    except Exception:
        return 0.0


def gpu_memory_reserved_mb() -> float:
    """GPU memory reserved by the PyTorch caching allocator (MB)."""
    if not _TORCH or not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.memory_reserved()) / (1024 ** 2)
    except Exception:
        return 0.0


def gpu_max_memory_allocated_mb() -> float:
    """Peak GPU memory allocated since last reset (MB)."""
    if not _TORCH or not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    except Exception:
        return 0.0


def reset_gpu_peak_memory() -> None:
    """Reset the peak memory counter for a fresh measurement window."""
    if _TORCH and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def gpu_utilization_percent() -> float:
    """
    GPU utilisation % via nvidia-smi (subprocess).
    Returns 0.0 if nvidia-smi is not available.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines:
                return float(lines[0].strip())
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Hardware snapshot
# ---------------------------------------------------------------------------

class HardwareMonitor:
    """
    Lightweight context-manager / snapshot utility for tracking
    memory and timing during a pipeline stage.

    Usage
    -----
    >>> with HardwareMonitor("training_original") as mon:
    ...     train_model(...)
    >>> print(mon.summary())
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.t_start: float = 0.0
        self.t_end: float = 0.0
        self.cpu_mb_start: float = 0.0
        self.cpu_mb_end: float = 0.0
        self.cpu_mb_peak: float = 0.0
        self.gpu_mb_start: float = 0.0
        self.gpu_mb_end: float = 0.0
        self.gpu_mb_peak: float = 0.0
        self.gpu_util_start: float = 0.0
        self.gpu_util_end: float = 0.0

    def __enter__(self) -> "HardwareMonitor":
        reset_gpu_peak_memory()
        self.t_start = time.perf_counter()
        self.cpu_mb_start = rss_mb()
        self.gpu_mb_start = gpu_memory_allocated_mb()
        self.gpu_util_start = gpu_utilization_percent()
        return self

    def __exit__(self, *_) -> None:
        self.t_end = time.perf_counter()
        self.cpu_mb_end = rss_mb()
        self.gpu_mb_end = gpu_memory_allocated_mb()
        self.gpu_mb_peak = gpu_max_memory_allocated_mb()
        self.gpu_util_end = gpu_utilization_percent()
        self.cpu_mb_peak = max(self.cpu_mb_start, self.cpu_mb_end)

    @property
    def elapsed_s(self) -> float:
        return self.t_end - self.t_start

    def summary(self) -> Dict:
        return {
            "label": self.label,
            "elapsed_s": round(self.elapsed_s, 4),
            "cpu_rss_start_MB": round(self.cpu_mb_start, 2),
            "cpu_rss_end_MB": round(self.cpu_mb_end, 2),
            "cpu_rss_delta_MB": round(self.cpu_mb_end - self.cpu_mb_start, 2),
            "gpu_alloc_start_MB": round(self.gpu_mb_start, 2),
            "gpu_alloc_end_MB": round(self.gpu_mb_end, 2),
            "gpu_peak_MB": round(self.gpu_mb_peak, 2),
            "gpu_util_start_pct": round(self.gpu_util_start, 1),
            "gpu_util_end_pct": round(self.gpu_util_end, 1),
        }


# ---------------------------------------------------------------------------
# Full hardware snapshot
# ---------------------------------------------------------------------------

def collect_hardware_info() -> Dict:
    """Return a combined hardware snapshot (CPU + GPU + RAM)."""
    info = {**get_cpu_info(), **get_gpu_info()}
    info["current_cpu_rss_MB"] = round(rss_mb(), 2)
    info["current_gpu_alloc_MB"] = round(gpu_memory_allocated_mb(), 2)
    info["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return info

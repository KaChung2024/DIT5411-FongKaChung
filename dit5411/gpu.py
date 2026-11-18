"""GPU setup for an RTX 5080 (Blackwell, 16 GB, compute capability 12.0)."""

from __future__ import annotations

import subprocess
import sys

import tensorflow as tf


def _version_tuple(text: str) -> tuple[int, int]:
    parts = []
    for piece in text.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts[:2]) if parts else (0, 0)


def configure_runtime(use_mixed_precision: bool = True) -> list:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth cannot be changed after the device has been used.
            pass
    if use_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    return gpus


def describe_devices(gpus: list) -> None:
    print(f"Python {sys.version.split()[0]}")
    print(f"TensorFlow {tf.__version__}")
    print(f"Policy: {tf.keras.mixed_precision.global_policy().name}")
    if not gpus:
        print("No GPU visible. Training will use the CPU and will not hit the 1 minute target.")
        print("On the RTX 5080 machine install the CUDA build: pip install -r requirements-gpu.txt")
        return
    for gpu in gpus:
        details = {}
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:
            details = {}
        name = details.get("device_name", gpu.name)
        capability = details.get("compute_capability")
        print(f"GPU: {name}  compute capability: {capability}")
        if isinstance(capability, (tuple, list)) and tuple(capability) >= (12, 0):
            version = _version_tuple(tf.__version__)
            print("Blackwell GPU detected (RTX 50 series). Mixed precision + XLA compile for this device.")
            if version < (2, 21):
                print(
                    "Warning: TensorFlow older than 2.21 often JIT-compiles Blackwell kernels "
                    "and leaves Tensor Cores idle. Prefer tensorflow[and-cuda] 2.21 or newer, "
                    "or an NVIDIA NGC TensorFlow container. Driver must be 570 or newer "
                    "(580-open recommended)."
                )


def gpu_total_bytes(index: int = 0) -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(output.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return 16 * 1024**3


def read_gpu_sample(index: int = 0) -> tuple[float, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        util, power = output.strip().splitlines()[0].split(",")
        return float(util), float(power)
    except Exception:
        return None

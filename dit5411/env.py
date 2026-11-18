"""Process environment for TensorFlow. Must run before TensorFlow is imported."""

import os
import sys

REQUIRED_PYTHON = (3, 13)


def require_python() -> None:
    """Python 3.13 inside Ubuntu on WSL2. Native Windows cannot drive the RTX 5080."""
    if sys.platform == "win32":
        raise SystemExit(
            "Run this in Ubuntu on WSL2, not in Windows Python.\n"
            "TensorFlow cannot use the RTX 5080 from native Windows.\n"
            "Open Ubuntu from Windows Terminal, then:\n"
            "  cd ~/DIT5411_chinese\n"
            "  python3.13 -m venv .venv && source .venv/bin/activate\n"
            "Keep the project and data/packed in the Ubuntu home directory, not under /mnt/c."
        )
    if sys.version_info[:2] == REQUIRED_PYTHON:
        return
    running = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(
        f"This project uses Python 3.13 (this interpreter is {running}).\n"
        "Inside Ubuntu on WSL2:\n"
        "  python3.13 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -U pip\n"
        "  pip install -r requirements-gpu.txt"
    )


def warn_if_slow_mount(path) -> None:
    """/mnt/c is the Windows disk. An 11 GB sequential read from there misses the epoch target."""
    try:
        resolved = os.path.realpath(path)
    except OSError:
        resolved = os.fspath(path)
    parts = resolved.split("/")
    if len(parts) < 3 or parts[1] != "mnt" or parts[2] in {"wsl", "wslg"} or len(parts[2]) != 1:
        return
    print(
        f"Warning: {resolved} is on the Windows drive.\n"
        "Copy the project and the packed dataset into the Ubuntu home directory "
        "(for example ~/DIT5411_chinese). Reading train.bin from /mnt/c will not "
        "finish an epoch in 60 seconds."
    )


def configure_env() -> None:
    """Prefer fast GPU defaults without overriding variables already set by the user."""
    require_python()
    warn_if_slow_mount(os.getcwd())
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Async allocator avoids sync stalls when the batch size fills the 16 GB card.
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")
    # Private GPU threads overlap the packed-file reader with compute.
    os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
    os.environ.setdefault("TF_GPU_THREAD_COUNT", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

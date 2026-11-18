"""Packed training records.

Each record is raw uint8 pixels, row-major, then a little-endian uint16 class
index. Sequential reads of this layout are what keep an epoch under a minute:
the training loop never opens the original image files.
"""

from __future__ import annotations

import numpy as np

LABEL_BYTES = 2


def record_bytes(img_size: int) -> int:
    return img_size * img_size + LABEL_BYTES


def pack_record(image: np.ndarray, label: int) -> bytes:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError(f"expected HxW grayscale, got shape {image.shape}")
    if not 0 <= int(label) <= 65535:
        raise ValueError(f"label {label} does not fit in uint16")
    return image.reshape(-1).tobytes() + int(label).to_bytes(LABEL_BYTES, "little", signed=False)


def unpack_record(blob: bytes, img_size: int) -> tuple[np.ndarray, int]:
    hw = img_size * img_size
    if len(blob) != hw + LABEL_BYTES:
        raise ValueError(f"record length {len(blob)} != {hw + LABEL_BYTES}")
    image = np.frombuffer(blob[:hw], dtype=np.uint8).reshape(img_size, img_size).copy()
    label = int.from_bytes(blob[hw : hw + LABEL_BYTES], "little", signed=False)
    return image, label

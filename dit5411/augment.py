"""Geometric augmentation with OpenCV.

The assignment asks for augmented training images (rotation, shearing, scaling).
Transforms run once, offline, at a working resolution twice the network input,
then the result is resampled to the training size. Doing this before training
keeps the GPU loop free of image decoding.
"""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

# Ranges below are mild on purpose. A large rotation turns one Chinese
# character into a different one, which would poison the label.
ROTATE_DEG = (8.0, 16.0)
SHEAR = (0.12, 0.28)
SCALE_UP = (1.08, 1.22)
COMBINED_DEG = (5.0, 12.0)
COMBINED_SHEAR = (0.08, 0.18)
COMBINED_SCALE = (0.88, 1.12)
SPECS = ("rotate", "shear", "scale", "combined")

AUGMENTATION_SPEC = {
    "library": "OpenCV cv2.getRotationMatrix2D + cv2.warpAffine",
    "reference": "https://github.com/darshitajain/Introduction-to-OpenCV",
    "rotate_degrees": list(ROTATE_DEG),
    "shear": list(SHEAR),
    "scale_up": list(SCALE_UP),
    "scale_down": "reciprocal of scale_up, so both zoom-in and zoom-out",
    "combined_degrees": list(COMBINED_DEG),
    "combined_shear": list(COMBINED_SHEAR),
    "combined_scale": list(COMBINED_SCALE),
    "direction": "each sign is chosen from the sample seed, so both directions occur",
    "border": "constant fill with the median colour of the four corners",
    "variants_per_original_when_40_sources": 4,
    "originals_kept": True,
}


def imread_gray(path: str) -> np.ndarray | None:
    """Read a grayscale image. cv2.imread fails on non-ASCII paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    return image


def _background(image: np.ndarray) -> int:
    strips = (
        image[0, :8],
        image[-1, :8],
        image[:8, 0],
        image[:8, -1],
    )
    return int(np.median(np.concatenate([s.reshape(-1) for s in strips])))


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(image, dtype=np.uint8)
    interp = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_LINEAR
    out = cv2.resize(image, (size, size), interpolation=interp)
    return np.ascontiguousarray(out, dtype=np.uint8)


def apply_geom(image: np.ndarray, angle: float = 0.0, shear: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Rotate, shear and scale around the image centre. Angle is in degrees."""
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    rotate = cv2.getRotationMatrix2D(centre, float(angle), float(scale))
    shear_m = np.array(
        [[1.0, float(shear), -float(shear) * centre[0]], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotate_m = np.vstack([rotate, [0.0, 0.0, 1.0]])
    composed = (shear_m @ rotate_m)[:2].astype(np.float32)
    warped = cv2.warpAffine(
        image,
        composed,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(_background(image)),
    )
    return np.ascontiguousarray(warped, dtype=np.uint8)


def make_rng(class_index: int, sample_index: int, aug_index: int, base_seed: int = 2026) -> np.random.Generator:
    seed = (base_seed + class_index * 100003 + sample_index * 9176 + aug_index * 97) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def sample_params(spec: str, rng: np.random.Generator) -> dict[str, float]:
    sign = float(rng.choice([-1.0, 1.0]))
    if spec == "rotate":
        return {"angle": sign * float(rng.uniform(*ROTATE_DEG)), "shear": 0.0, "scale": 1.0}
    if spec == "shear":
        return {
            "angle": float(rng.uniform(-3.0, 3.0)),
            "shear": sign * float(rng.uniform(*SHEAR)),
            "scale": 1.0,
        }
    if spec == "scale":
        factor = float(rng.uniform(*SCALE_UP))
        if sign < 0:
            factor = 1.0 / factor
        return {"angle": float(rng.uniform(-3.0, 3.0)), "shear": 0.0, "scale": factor}
    if spec == "combined":
        return {
            "angle": sign * float(rng.uniform(*COMBINED_DEG)),
            "shear": float(rng.choice([-1.0, 1.0])) * float(rng.uniform(*COMBINED_SHEAR)),
            "scale": float(rng.uniform(*COMBINED_SCALE)),
        }
    raise ValueError(spec)


def augment_image(
    image: np.ndarray,
    img_size: int,
    work_size: int,
    params: Mapping[str, float] | None = None,
) -> np.ndarray:
    canvas = resize_square(image, work_size)
    if params is not None:
        canvas = apply_geom(canvas, **params)
    return resize_square(canvas, img_size)


def extras_per_image(n_originals: int, target_total: int) -> list[int]:
    """How many augmented copies to add for each original so the class reaches `target_total`."""
    if n_originals <= 0:
        raise ValueError("no original images")
    extras = max(0, target_total - n_originals)
    base, rem = divmod(extras, n_originals)
    return [base + (1 if i < rem else 0) for i in range(n_originals)]


def render_method_figure(path: str, img_size: int = 64) -> None:
    """Write a contact sheet of the four transforms on a drawn stroke image."""
    glyph = np.full((160, 160), 255, np.uint8)
    cv2.line(glyph, (36, 24), (36, 140), 0, 5)
    cv2.line(glyph, (36, 36), (124, 36), 0, 5)
    cv2.line(glyph, (48, 78), (128, 58), 0, 4)
    cv2.rectangle(glyph, (72, 92), (132, 142), 0, 3)
    cv2.line(glyph, (78, 96), (126, 136), 0, 3)

    demos = [
        ("original", None),
        ("rotation", {"angle": 14.0, "shear": 0.0, "scale": 1.0}),
        ("shear", {"angle": 0.0, "shear": 0.25, "scale": 1.0}),
        ("scale", {"angle": 0.0, "shear": 0.0, "scale": 1.18}),
        ("combined", {"angle": -10.0, "shear": -0.16, "scale": 0.9}),
    ]
    cell = img_size * 3
    gap = 8
    header = 28
    canvas = np.full((header + cell + 8, 8 + len(demos) * (cell + gap)), 255, np.uint8)
    for col, (name, params) in enumerate(demos):
        tile = augment_image(glyph, img_size, img_size * 2, params)
        view = cv2.resize(tile, (cell, cell), interpolation=cv2.INTER_NEAREST)
        x0 = 8 + col * (cell + gap)
        canvas[header : header + cell, x0 : x0 + cell] = view
        cv2.putText(canvas, name, (x0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("failed to encode the augmentation figure")
    buf.tofile(path)

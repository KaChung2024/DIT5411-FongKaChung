#!/usr/bin/env python3
"""Predict the character in one grayscale or colour image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dit5411.env import configure_env

configure_env()

try:
    import tensorflow as tf

    from dit5411.augment import augment_image, imread_gray
    from dit5411.gpu import configure_runtime
    from dit5411.losses import SparseLabelSmoothingCE
    from dit5411.models import build_model
    from dit5411.schedule import WarmupCosine
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. On the RTX 5080 machine run:\n  pip install -r requirements-gpu.txt"
    ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/fast_resnet"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    characters = json.loads((args.run / "characters.json").read_text(encoding="utf-8"))
    image = imread_gray(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read {args.image}")
    work_size = int(config.get("work_size") or int(config["img_size"]) * 2)
    tile = augment_image(image, int(config["img_size"]), work_size, None)
    configure_runtime(True)
    weights = args.run / "best.keras"
    loaded = tf.keras.models.load_model(
        weights,
        custom_objects={"SparseLabelSmoothingCE": SparseLabelSmoothingCE, "WarmupCosine": WarmupCosine},
        compile=False,
    )
    model = build_model(config["model"], int(config["img_size"]), int(config["num_classes"]))
    model.set_weights(loaded.get_weights())
    batch = tf.constant(tile.reshape(1, tile.shape[0], tile.shape[1], 1))
    logits = model(batch, training=False)
    k = min(args.topk, int(config["num_classes"]))
    _values, indices = tf.math.top_k(logits, k=k)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    indices = indices.numpy()[0]
    print(f"Top {k} for {args.image}:")
    for rank, index in enumerate(indices, start=1):
        name = characters[index] if index < len(characters) else str(index)
        print(f"  {rank}. {name}  {probs[index] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

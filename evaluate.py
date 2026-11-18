#!/usr/bin/env python3
"""Score a saved model on every held-out test image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dit5411.env import configure_env

configure_env()

try:
    import tensorflow as tf

    from dit5411.data import load_meta
    from dit5411.gpu import configure_runtime
    from dit5411.losses import SparseLabelSmoothingCE
    from dit5411.models import build_model
    from dit5411.schedule import WarmupCosine
    from dit5411.testing import evaluate_model
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. On the RTX 5080 machine run:\n  pip install -r requirements-gpu.txt"
    ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/fast_resnet"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="Defaults to <run>/best.keras")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data_dir"])
    weights = args.weights or (args.run / "best.keras")
    meta = load_meta(data_dir)
    configure_runtime(True)
    loaded = tf.keras.models.load_model(
        weights,
        custom_objects={"SparseLabelSmoothingCE": SparseLabelSmoothingCE, "WarmupCosine": WarmupCosine},
        compile=False,
    )
    model = build_model(config["model"], int(meta["img_size"]), int(meta["num_classes"]))
    model.set_weights(loaded.get_weights())
    metrics = evaluate_model(model, data_dir, meta, int(config["batch_size"]))
    print(
        f"TEST RECOGNITION RATE ({config['model']}): {metrics['recognition_rate_percent']:.2f}%  "
        f"top-5 {metrics['test_top5'] * 100:.2f}%  on {metrics['test_samples']:,} images"
    )
    (args.run / "result.json").write_text(
        json.dumps({**config, **metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

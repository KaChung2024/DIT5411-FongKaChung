#!/usr/bin/env python3
"""Train a Chinese handwriting model on the packed dataset.

Set the process environment before TensorFlow is imported.
"""

from __future__ import annotations

import argparse
import sys

from dit5411.env import configure_env

configure_env()

try:
    from dit5411.models import MODEL_NAMES
    from dit5411.trainer import benchmark, check_gpu, summary_only, train_all, train_one
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. On the RTX 5080 machine run:\n"
        "  pip install -r requirements-gpu.txt\n"
        f"Import error: {exc}"
    ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fast_resnet", choices=[*MODEL_NAMES, "all"])
    parser.add_argument("--data-dir", default="data/packed")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=0, help="0 picks a batch that fills the GPU.")
    parser.add_argument("--lr", type=float, default=0.0, help="0 scales 1e-3 by sqrt(batch/256), capped at 4e-3.")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--steps-per-execution", type=int, default=16, help="Steps fused into one GPU call.")
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--limit-seconds", type=float, default=60.0)
    parser.add_argument("--early-stop", type=int, default=0, help="Patience in epochs. 0 disables early stopping.")
    parser.add_argument("--no-xla", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Time the model on synthetic batches. No dataset required.")
    parser.add_argument("--benchmark-steps", type=int, default=30)
    parser.add_argument("--benchmark-samples", type=int, default=13065 * 200)
    parser.add_argument("--img-size", type=int, default=64, help="Used by --benchmark and --summary-only.")
    parser.add_argument("--num-classes", type=int, default=13065, help="Used by --benchmark and --summary-only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_gpu:
        return check_gpu()
    if args.summary_only:
        names = MODEL_NAMES if args.model == "all" else (args.model,)
        for name in names:
            summary_only(name, args.img_size, args.num_classes)
        return 0
    if args.benchmark:
        name = "fast_resnet" if args.model == "all" else args.model
        return benchmark(
            name,
            args.img_size,
            args.num_classes,
            args.batch_size,
            args.benchmark_samples,
            use_xla=not args.no_xla,
            steps=args.benchmark_steps,
        )
    if args.model == "all":
        return train_all(args)
    train_one(args, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create the augmented train/test packs from the Traditional Chinese Handwriting Dataset."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

from dit5411.env import require_python

require_python()

import cv2
import numpy as np

from dit5411.augment import augment_image, make_rng, render_method_figure, sample_params
from dit5411.prepare import discover_classes, run_prepare, sample_index, split_files
from dit5411.records import pack_record, record_bytes, unpack_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Path to cleaned_data (one folder per character).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/packed"))
    parser.add_argument("--img-size", type=int, default=64, help="Network input. 64 keeps an RTX 5080 epoch under a minute.")
    parser.add_argument("--train-count", type=int, default=40, help="First N originals of each character go to training.")
    parser.add_argument("--target-per-class", type=int, default=200, help="Training images per character after augmentation.")
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count() or 1))
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-any-classes", action="store_true", help="Pack a subset. The assignment set has 13,065 characters.")
    parser.add_argument("--self-test", action="store_true", help="Run a tiny local test of the split, augmentation and pack.")
    return parser.parse_args(argv)


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to write {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def self_test() -> int:
    blob = pack_record(np.arange(16, dtype=np.uint8).reshape(4, 4), 13064)
    image, label = unpack_record(blob, 4)
    assert label == 13064, label
    assert image.shape == (4, 4)
    assert int(image[0, 1]) == 1

    files = sorted([Path(f"{i}.png") for i in (10, 1, 2, 0, 4, 3)], key=sample_index)
    train, test, warning = split_files(files, 4)
    assert warning is None
    assert [path.name for path in train] == ["0.png", "1.png", "2.png", "3.png"]
    assert [path.name for path in test] == ["4.png", "10.png"]

    glyph = np.full((80, 80), 255, np.uint8)
    cv2.circle(glyph, (40, 40), 18, 0, 2)
    first = sample_params("rotate", make_rng(1, 2, 3))
    second = sample_params("rotate", make_rng(1, 2, 3))
    left = augment_image(glyph, 32, 64, first)
    right = augment_image(glyph, 32, 64, second)
    plain = augment_image(glyph, 32, 64, None)
    assert left.shape == (32, 32)
    assert np.array_equal(left, right)
    assert not np.array_equal(left, plain)

    root = Path("data/_selftest")
    if root.exists():
        shutil.rmtree(root)
    source = root / "cleaned_data"
    for class_id in range(3):
        for index in (0, 1, 2, 3, 4, 10):
            canvas = np.full((36, 36), 240, np.uint8)
            cv2.line(canvas, (4, 6 + class_id), (30, 28), 0, 2)
            _write_png(source / f"c{class_id}" / f"{index}.png", canvas)
    found = discover_classes(source)
    assert list(found) == ["c0", "c1", "c2"], list(found)
    assert [path.name for path in found["c0"]] == ["0.png", "1.png", "2.png", "3.png", "4.png", "10.png"]

    meta = run_prepare(
        source,
        root / "packed",
        img_size=32,
        train_count=4,
        target_per_class=8,
        workers=2,
        shards=4,
        overwrite=True,
        allow_any_classes=True,
        publish_preview=False,
    )
    record = record_bytes(32)
    train_path = root / "packed" / "train.bin"
    test_path = root / "packed" / "test.bin"
    assert train_path.stat().st_size == 3 * 8 * record
    assert test_path.stat().st_size == 3 * 2 * record
    assert meta["train_samples"] == 24
    assert meta["test_samples"] == 6
    raw = np.fromfile(train_path, dtype=np.uint8).reshape(-1, record)
    labels = [int.from_bytes(row[-2:].tobytes(), "little") for row in raw]
    assert sorted(labels) == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    render_method_figure(str(docs / "augmentation_examples.jpg"), 64)
    shutil.rmtree(root)
    print("Self-test passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.data_dir is None:
        raise SystemExit("Pass --data-dir, or run --self-test.")
    if args.target_per_class < 200 and not args.allow_any_classes:
        raise SystemExit("The assignment asks for at least 200 training images per character.")
    run_prepare(
        args.data_dir,
        args.out_dir,
        img_size=args.img_size,
        train_count=args.train_count,
        target_per_class=args.target_per_class,
        workers=args.workers,
        shards=args.shards,
        seed=args.seed,
        overwrite=args.overwrite,
        allow_any_classes=args.allow_any_classes,
        publish_preview=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())

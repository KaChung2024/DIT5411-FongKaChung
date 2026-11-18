"""Split the handwriting folders, augment the training images, pack both splits.

Layout accepted from the AI-FREE dataset (either one):

- ``cleaned_data/<character>/*.jpg``
- a flat folder of ``<character>_<index>.jpg``

The first ``train_count`` files of each character (numeric filename order) are
the training originals. Every remaining file is the test set and is not
augmented. Training originals are kept, then OpenCV variants are added until
each character has ``target_per_class`` training images.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dit5411.augment import (
    AUGMENTATION_SPEC,
    SPECS,
    augment_image,
    extras_per_image,
    imread_gray,
    make_rng,
    render_method_figure,
    sample_params,
)
from dit5411.env import warn_if_slow_mount
from dit5411.records import pack_record, record_bytes

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".ppm"}
EXPECTED_CLASSES = 13065


def sample_index(path: Path) -> tuple:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, path.name)


def discover_classes(data_dir: Path) -> dict[str, list[Path]]:
    data_dir = _resolve_data_dir(data_dir)
    entries = [p for p in data_dir.iterdir() if not p.name.startswith(".")]
    n_dir = sum(p.is_dir() for p in entries)
    n_img = sum(p.is_file() and p.suffix.lower() in IMG_EXTS for p in entries)
    classes: dict[str, list[Path]] = {}
    if n_dir > 0 and n_dir >= n_img:
        for folder in entries:
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            files.sort(key=sample_index)
            if files:
                classes[folder.name] = files
    else:
        groups: dict[str, list[Path]] = defaultdict(list)
        for path in entries:
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                groups[path.stem.split("_")[0]].append(path)
        for name, files in groups.items():
            files.sort(key=sample_index)
            classes[name] = files
    if not classes:
        raise FileNotFoundError(
            f"No character images under {data_dir}. Point --data-dir at cleaned_data "
            "(one folder per character) or at a flat folder of <character>_<index>.jpg files."
        )
    return dict(sorted(classes.items(), key=lambda item: item[0]))


def _resolve_data_dir(data_dir: Path) -> Path:
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if _looks_like_dataset(data_dir):
        return data_dir
    children = [p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    candidates = [p for p in children if _looks_like_dataset(p)]
    if len(candidates) == 1:
        print(f"Using dataset folder {candidates[0]}")
        return candidates[0]
    return data_dir


def _looks_like_dataset(path: Path) -> bool:
    try:
        entries = [p for p in path.iterdir() if not p.name.startswith(".")]
    except OSError:
        return False
    if any(p.is_dir() for p in entries):
        return True
    return sum(p.suffix.lower() in IMG_EXTS for p in entries if p.is_file()) > 20


def split_files(files: list[Path], train_count: int) -> tuple[list[Path], list[Path], str | None]:
    """Return (train, test, warning). First ``train_count`` files train, the rest test."""
    files = list(files)
    n = len(files)
    if n == 0:
        raise ValueError("empty class")
    if n > train_count:
        return files[:train_count], files[train_count:], None
    if n == 1:
        return files, list(files), "only 1 image; the same image is used for train and test"
    return files[:-1], files[-1:], f"only {n} images; holding out the last one for test"


def _init_worker() -> None:
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def process_class(job: dict) -> dict:
    """Build one character's packed records. Must stay picklable for Windows spawn."""
    _init_worker()
    img_size = int(job["img_size"])
    work_size = int(job["work_size"])
    class_index = int(job["class_index"])
    train_files = job["train_files"]
    test_files = job["test_files"]
    target = int(job["target_per_class"])
    tmp_dir = Path(job["tmp_dir"])

    originals = []
    skipped = 0
    for path in train_files:
        image = imread_gray(path)
        if image is None:
            skipped += 1
            continue
        originals.append(image)
    if not originals:
        raise RuntimeError(f"no readable training images for class {job['class_name']!r}")

    counts = extras_per_image(len(originals), target)
    train_path = tmp_dir / f"train_{class_index:05d}.bin"
    with train_path.open("wb") as handle:
        for sample_index, image in enumerate(originals):
            handle.write(pack_record(augment_image(image, img_size, work_size, None), class_index))
            for aug_index in range(counts[sample_index]):
                spec = SPECS[aug_index % len(SPECS)]
                rng = make_rng(class_index, sample_index, aug_index, int(job["seed"]))
                params = sample_params(spec, rng)
                handle.write(pack_record(augment_image(image, img_size, work_size, params), class_index))

    test_path = tmp_dir / f"test_{class_index:05d}.bin"
    test_count = 0
    with test_path.open("wb") as handle:
        for path in test_files:
            image = imread_gray(path)
            if image is None:
                skipped += 1
                continue
            handle.write(pack_record(augment_image(image, img_size, work_size, None), class_index))
            test_count += 1

    return {
        "class_index": class_index,
        "class_name": job["class_name"],
        "train_path": str(train_path),
        "train_count": len(originals) + sum(counts),
        "test_path": str(test_path),
        "test_count": test_count,
        "n_source": len(job["source_files"]),
        "n_train_originals": len(originals),
        "skipped": skipped,
        "warning": job.get("warning"),
    }


def _scatter(class_results: list[dict], key_path: str, key_count: str, out_path: Path, record_size: int, shards: int, seed: int) -> int:
    shard_dir = out_path.parent / f"_shards_{out_path.stem}"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    handles = [(shard_dir / f"shard_{i:03d}.bin").open("wb") for i in range(shards)]
    buffers = [bytearray() for _ in range(shards)]
    flush_at = 4 * 1024 * 1024
    total = 0
    try:
        for result in class_results:
            path = Path(result[key_path])
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size == 0:
                continue
            if raw.size % record_size != 0:
                raise RuntimeError(f"{path} is not a whole number of records")
            rows = raw.reshape(-1, record_size)
            shard_id = (int(result["class_index"]) * 17 + np.arange(rows.shape[0])) % shards
            for shard in range(shards):
                chunk = rows[shard_id == shard]
                if chunk.size == 0:
                    continue
                buffers[shard].extend(np.ascontiguousarray(chunk).tobytes())
                if len(buffers[shard]) >= flush_at:
                    handles[shard].write(buffers[shard])
                    buffers[shard].clear()
            total += int(result[key_count])
            path.unlink(missing_ok=True)
        for shard, pending in enumerate(buffers):
            if pending:
                handles[shard].write(pending)
    finally:
        for handle in handles:
            handle.close()

    rng = np.random.default_rng(seed)
    with out_path.open("wb") as dest:
        for shard in range(shards):
            shard_path = shard_dir / f"shard_{shard:03d}.bin"
            raw = np.fromfile(shard_path, dtype=np.uint8)
            if raw.size:
                rows = raw.reshape(-1, record_size)
                rng.shuffle(rows)
                dest.write(np.ascontiguousarray(rows).tobytes())
            shard_path.unlink(missing_ok=True)
    shard_dir.rmdir()
    written = out_path.stat().st_size // record_size
    if written != total:
        raise RuntimeError(f"{out_path.name}: wrote {written} records, expected {total}")
    return written


def save_real_previews(classes: dict, names: list[str], img_size: int, work_size: int, dest: Path, seed: int) -> None:
    """Write a contact sheet of real characters after the same four transforms."""
    rows = []
    labels = []
    for class_index, name in list(enumerate(names))[:4]:
        source = None
        for path in classes[name][:40]:
            source = imread_gray(str(path))
            if source is not None:
                break
        if source is None:
            continue
        tiles = [augment_image(source, img_size, work_size, None)]
        for aug_index, spec in enumerate(SPECS):
            params = sample_params(spec, make_rng(class_index, 0, aug_index, seed))
            tiles.append(augment_image(source, img_size, work_size, params))
        rows.append(tiles)
        labels.append(name)
    if not rows:
        return
    dest.mkdir(parents=True, exist_ok=True)
    cell = img_size * 3
    gap = 6
    header = 8
    canvas = np.full((header + len(rows) * (cell + gap), 8 + 5 * (cell + gap)), 255, np.uint8)
    for row_index, tiles in enumerate(rows):
        for col_index, tile in enumerate(tiles):
            view = cv2.resize(tile, (cell, cell), interpolation=cv2.INTER_NEAREST)
            y0 = header + row_index * (cell + gap)
            x0 = 8 + col_index * (cell + gap)
            canvas[y0 : y0 + cell, x0 : x0 + cell] = view
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return
    encoded.tofile(str(dest / "real_augmentation_examples.jpg"))
    (dest / "real_augmentation_labels.txt").write_text(
        "Columns: original, rotation, shear, scale, combined\n"
        + "\n".join(f"row {i}: {name}" for i, name in enumerate(labels))
        + "\n",
        encoding="utf-8",
    )


def run_prepare(
    data_dir: Path,
    out_dir: Path,
    img_size: int = 64,
    train_count: int = 40,
    target_per_class: int = 200,
    workers: int = 8,
    shards: int = 64,
    seed: int = 2026,
    overwrite: bool = False,
    allow_any_classes: bool = False,
    publish_preview: bool = True,
) -> dict:
    warn_if_slow_mount(data_dir)
    warn_if_slow_mount(out_dir)
    if img_size < 16 or img_size % 8 != 0:
        raise ValueError("--img-size must be a multiple of 8 and at least 16 (Tensor Core friendly)")
    if target_per_class < 1:
        raise ValueError("--target-per-class must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"{meta_path} already exists. Pass --overwrite to rebuild it.")

    classes = discover_classes(Path(data_dir))
    names = list(classes)
    if len(names) != EXPECTED_CLASSES and not allow_any_classes:
        raise SystemExit(
            f"Found {len(names)} characters, the assignment expects {EXPECTED_CLASSES}. "
            "Point --data-dir at the full cleaned_data folder (13,065 characters), "
            "or pass --allow-any-classes for a subset experiment."
        )

    work_size = max(img_size * 2, 96)
    record_size = record_bytes(img_size)
    estimate = len(names) * target_per_class * record_size
    free = shutil.disk_usage(out_dir).free
    print(
        f"{len(names)} characters. Packing about {len(names) * target_per_class:,} "
        f"training images at {img_size}x{img_size} ({estimate / 1024**3:.1f} GB). "
        f"Free space: {free / 1024**3:.1f} GB."
    )
    if free < estimate * 2:
        print("Warning: free disk is tight. The pack step needs room for a temporary copy.")

    tmp_dir = out_dir / "_tmp_records"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    jobs = []
    short_classes = 0
    for index, name in enumerate(names):
        train_files, test_files, warning = split_files(classes[name], train_count)
        if warning:
            short_classes += 1
        jobs.append(
            {
                "class_index": index,
                "class_name": name,
                "source_files": [str(p) for p in classes[name]],
                "train_files": [str(p) for p in train_files],
                "test_files": [str(p) for p in test_files],
                "warning": warning,
                "img_size": img_size,
                "work_size": work_size,
                "target_per_class": target_per_class,
                "tmp_dir": str(tmp_dir),
                "seed": seed,
            }
        )

    worker_count = max(1, min(workers, len(jobs)))
    print(f"Augmenting with {worker_count} workers (OpenCV rotation, shear, scale)...")
    if worker_count == 1:
        results = [process_class(job) for job in tqdm(jobs)]
    else:
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker) as pool:
            results = list(tqdm(pool.map(process_class, jobs, chunksize=4), total=len(jobs)))
    results.sort(key=lambda item: item["class_index"])

    print("Shuffling packed records into train.bin and test.bin ...")
    train_samples = _scatter(results, "train_path", "train_count", out_dir / "train.bin", record_size, shards, seed)
    test_samples = _scatter(results, "test_path", "test_count", out_dir / "test.bin", record_size, max(1, min(shards, 8)), seed + 1)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    preview = out_dir / "preview.jpg"
    render_method_figure(str(preview), img_size)
    save_real_previews(classes, names, img_size, work_size, out_dir, seed)
    if publish_preview:
        docs = Path("docs")
        docs.mkdir(exist_ok=True)
        shutil.copyfile(preview, docs / "augmentation_examples.jpg")
        real_preview = out_dir / "real_augmentation_examples.jpg"
        if real_preview.is_file():
            shutil.copyfile(real_preview, docs / "real_augmentation_examples.jpg")
            shutil.copyfile(out_dir / "real_augmentation_labels.txt", docs / "real_augmentation_labels.txt")

    train_originals = [item["n_train_originals"] for item in results]
    test_counts = [item["test_count"] for item in results]
    meta = {
        "img_size": img_size,
        "work_size": work_size,
        "num_classes": len(names),
        "characters": names,
        "record_bytes": record_size,
        "train_bin": "train.bin",
        "test_bin": "test.bin",
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_count_requested": train_count,
        "target_per_class": target_per_class,
        "train_originals_min": int(min(train_originals)),
        "train_originals_max": int(max(train_originals)),
        "test_per_class_min": int(min(test_counts)),
        "test_per_class_max": int(max(test_counts)),
        "classes_shorter_than_train_count": short_classes,
        "skipped_unreadable": int(sum(item["skipped"] for item in results)),
        "split": (
            f"Sort each character by the numeric filename. "
            f"The first {train_count} readable images are training originals; "
            f"every remaining image is the untouched test set. "
            f"Training originals are kept and OpenCV variants are added until "
            f"each character has {target_per_class} training images."
        ),
        "augmentation": AUGMENTATION_SPEC,
        "seed": seed,
        "dataset_source": "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Done. train={train_samples:,}  test={test_samples:,}  "
        f"classes={len(names)}  output neurons={len(names)}"
    )
    if short_classes:
        print(f"Warning: {short_classes} characters had {train_count} or fewer source images.")
    print(f"Wrote {meta_path}")
    return meta

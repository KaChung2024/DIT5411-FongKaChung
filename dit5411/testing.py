"""Full test-set recognition rate, including the last partial batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from dit5411.data import make_dataset


def evaluate_model(model, data_dir: str | Path, meta: dict, batch_size: int) -> dict:
    num_classes = int(meta["num_classes"])
    img_size = int(meta["img_size"])
    dataset = make_dataset(
        Path(data_dir) / meta["test_bin"],
        img_size=img_size,
        record_size=int(meta["record_bytes"]),
        batch_size=batch_size,
        training=False,
        drop_remainder=False,
        shuffle_buffer=0,
    )
    correct = 0
    top5_hits = 0
    total = 0
    per_correct = np.zeros(num_classes, dtype=np.int64)
    per_total = np.zeros(num_classes, dtype=np.int64)
    k = min(5, num_classes)

    for images, labels in dataset:
        labels_np = labels.numpy()
        count = int(labels_np.shape[0])
        if count != batch_size:
            images = tf.pad(images, [[0, batch_size - count], [0, 0], [0, 0], [0, 0]])
        logits = model(images, training=False)[:count]
        top = tf.math.top_k(logits, k=k).indices.numpy()
        hit = top[:, 0] == labels_np
        in_top5 = (top == labels_np[:, None]).any(axis=1)
        correct += int(hit.sum())
        top5_hits += int(in_top5.sum())
        total += count
        per_total += np.bincount(labels_np, minlength=num_classes)
        if hit.any():
            per_correct += np.bincount(labels_np[hit], minlength=num_classes)

    if total == 0:
        raise RuntimeError("The test set is empty.")
    accuracy = correct / total
    top5 = top5_hits / total
    present = per_total > 0
    per_accuracy = np.divide(per_correct, np.maximum(per_total, 1))
    order = np.argsort(np.where(present, per_accuracy, 2.0))
    characters = meta.get("characters") or []
    hardest = []
    for index in order[:20]:
        if per_total[index] == 0:
            continue
        name = characters[index] if index < len(characters) else str(index)
        hardest.append(
            {
                "character": name,
                "index": int(index),
                "accuracy": float(per_accuracy[index]),
                "correct": int(per_correct[index]),
                "total": int(per_total[index]),
            }
        )
    return {
        "test_samples": int(total),
        "test_accuracy": float(accuracy),
        "test_top5": float(top5),
        "recognition_rate_percent": float(accuracy * 100.0),
        "hardest_characters": hardest,
    }

"""tf.data reader for the packed uint8 records."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from dit5411.records import record_bytes


def load_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No {path}. Run prepare_dataset.py on cleaned_data before training."
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    expected = record_bytes(int(meta["img_size"]))
    if int(meta["record_bytes"]) != expected:
        raise RuntimeError("meta.json record size does not match img_size")
    return meta


def parse_batch(records, img_size: int):
    raw = tf.io.decode_raw(records, tf.uint8)
    hw = img_size * img_size
    images = tf.reshape(raw[:, :hw], (-1, img_size, img_size, 1))
    labels = tf.cast(raw[:, hw], tf.int32) + tf.bitwise.left_shift(
        tf.cast(raw[:, hw + 1], tf.int32), 8
    )
    return images, labels


def make_dataset(
    bin_path: str | Path,
    img_size: int,
    record_size: int,
    batch_size: int,
    training: bool,
    drop_remainder: bool,
    shuffle_buffer: int = 8192,
):
    dataset = tf.data.FixedLengthRecordDataset(
        str(bin_path),
        record_size,
        buffer_size=8 * 1024 * 1024,
    )
    if training and shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(
        batch_size,
        drop_remainder=drop_remainder,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda records: parse_batch(records, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    options = tf.data.Options()
    options.deterministic = False
    dataset = dataset.with_options(options)
    return dataset.prefetch(tf.data.AUTOTUNE)

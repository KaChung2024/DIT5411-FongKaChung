"""Train, benchmark, and compare the three networks."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import tensorflow as tf

from dit5411.data import load_meta, make_dataset
from dit5411.env import warn_if_slow_mount
from dit5411.gpu import configure_runtime, describe_devices, gpu_total_bytes, read_gpu_sample
from dit5411.losses import SparseLabelSmoothingCE
from dit5411.models import DESCRIPTIONS, MODEL_NAMES, build_model
from dit5411.schedule import WarmupCosine
from dit5411.testing import evaluate_model

DEFAULT_BATCH = {"mlp": 2048, "small_cnn": 1536, "fast_resnet": 1024}
FULL_TRAIN_SAMPLES = 13065 * 200


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self, steps: int, batch_size: int, limit_s: float):
        super().__init__()
        self.steps = steps
        self.batch_size = batch_size
        self.limit_s = limit_s
        self.times: list[float] = []
        self._t = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._t
        self.times.append(elapsed)
        rate = (self.steps * self.batch_size) / elapsed if elapsed else 0.0
        status = "within 60s" if elapsed <= self.limit_s else "OVER 60s"
        print(f"  epoch {epoch + 1}: {elapsed:.1f}s, {rate:,.0f} train images/s, {status}")


class GpuMonitor(tf.keras.callbacks.Callback):
    """Sample GPU use during the epoch. A full 5080 sits near 100% and up to 360 W."""

    def __init__(self):
        super().__init__()
        self.samples: list[tuple[float, float]] = []
        self._stop = False
        self._thread: threading.Thread | None = None

    def on_epoch_begin(self, epoch, logs=None):
        self.samples = []
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            sample = read_gpu_sample()
            if sample is not None:
                self.samples.append(sample)
            time.sleep(1.0)

    def on_epoch_end(self, epoch, logs=None):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return
        util = sum(item[0] for item in self.samples) / len(self.samples)
        power = sum(item[1] for item in self.samples) / len(self.samples)
        print(f"  GPU this epoch: average utilisation {util:.0f}%, average power {power:.0f} W")


class NanStop(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        loss = None if logs is None else logs.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            print("Loss became NaN. Re-run with a lower --lr (try 3e-4).")
            self.model.stop_training = True


def _optimizer(learning_rate):
    opt_cls = getattr(tf.keras.optimizers, "AdamW", tf.keras.optimizers.Adam)
    kwargs = {"learning_rate": learning_rate, "clipnorm": 1.0}
    if opt_cls.__name__ == "AdamW":
        kwargs["weight_decay"] = 1e-4
    return opt_cls(**kwargs)


def compile_model(model, learning_rate, num_classes: int, smoothing: float, steps_per_execution: int, jit: bool):
    model.compile(
        optimizer=_optimizer(learning_rate),
        loss=SparseLabelSmoothingCE(num_classes, smoothing),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
        jit_compile=jit,
        steps_per_execution=max(1, int(steps_per_execution)),
    )


def _peak_bytes() -> int | None:
    for name in ("GPU:0", "/GPU:0"):
        try:
            info = tf.config.experimental.get_memory_info(name)
        except Exception:
            continue
        peak = int(info.get("peak") or 0)
        if peak > 0:
            return peak
    return None


def _reset_peak() -> None:
    for name in ("GPU:0", "/GPU:0"):
        try:
            tf.config.experimental.reset_memory_stats(name)
            return
        except Exception:
            continue


def choose_batch_size(model_name: str, img_size: int, num_classes: int, cap: int = 4096) -> int:
    """Pick the largest safe batch from two probe steps so the 16 GB card stays full."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return 32
    fallback = min(cap, DEFAULT_BATCH.get(model_name, 1024))
    model = build_model(model_name, img_size, num_classes)
    compile_model(model, learning_rate=1e-3, num_classes=num_classes, smoothing=0.0, steps_per_execution=1, jit=False)

    def probe(batch: int) -> int | None:
        images = tf.zeros((batch, img_size, img_size, 1), dtype=tf.uint8)
        labels = tf.zeros((batch,), dtype=tf.int32)
        _reset_peak()
        try:
            model.train_on_batch(images, labels)
        except Exception as exc:
            text = str(exc).lower()
            if isinstance(exc, (tf.errors.ResourceExhaustedError, tf.errors.InternalError)) or "oom" in text or "resource exhausted" in text:
                return None
            raise
        return _peak_bytes()

    peak_lo = probe(64)
    peak_hi = probe(128) if peak_lo is not None else None
    tf.keras.backend.clear_session()
    if peak_lo is None:
        print(f"Batch probe ran out of memory. Using batch size {fallback}.")
        return fallback
    if peak_hi is None or peak_hi <= peak_lo:
        print(f"Could not measure how memory scales. Using batch size {fallback}.")
        return fallback

    per_sample = (peak_hi - peak_lo) / 64.0
    fixed = max(0.0, peak_lo - 64.0 * per_sample)
    usable = 0.82 * gpu_total_bytes() - fixed
    if per_sample <= 0 or usable <= 0:
        chosen = fallback
    else:
        chosen = int(usable / per_sample)
    chosen = max(128, min(cap, chosen))
    chosen -= chosen % 64
    print(
        f"Auto batch size {chosen} "
        f"(probe {peak_lo / 1024**2:.0f} MB at 64, {peak_hi / 1024**2:.0f} MB at 128, "
        f"GPU {gpu_total_bytes() / 1024**3:.1f} GB)."
    )
    return chosen


def _steady_seconds(times: list[float]) -> float | None:
    if not times:
        return None
    body = times[1:] if len(times) > 1 else times
    body = sorted(body)
    return float(body[len(body) // 2])


def check_gpu() -> int:
    gpus = configure_runtime(True)
    describe_devices(gpus)
    if not gpus:
        return 1
    print("Running a float16 matmul on GPU:0 ...")
    with tf.device("/GPU:0"):
        left = tf.random.normal((4096, 4096), dtype=tf.float16)
        product = tf.matmul(left, left)
        _ = product.numpy()
    print("GPU matmul succeeded.")
    return 0


def summary_only(model_name: str, img_size: int, num_classes: int) -> int:
    configure_runtime(False)
    model = build_model(model_name, img_size, num_classes)
    model.summary()
    print(DESCRIPTIONS[model_name])
    print(f"Output neurons: {model.output_shape[-1]}")
    print(f"Parameters: {model.count_params():,}")
    return 0


def benchmark(model_name: str, img_size: int, num_classes: int, batch_size: int, samples: int, use_xla: bool, steps: int) -> int:
    gpus = configure_runtime(True)
    describe_devices(gpus)
    if batch_size <= 0:
        batch_size = choose_batch_size(model_name, img_size, num_classes)
        configure_runtime(bool(gpus))
    model = build_model(model_name, img_size, num_classes)
    compile_model(
        model,
        learning_rate=1e-3,
        num_classes=num_classes,
        smoothing=0.05,
        steps_per_execution=min(16, steps),
        jit=use_xla and bool(gpus),
    )
    images = tf.cast(
        tf.random.uniform((batch_size, img_size, img_size, 1), maxval=256, dtype=tf.int32),
        tf.uint8,
    )
    labels = tf.random.uniform((batch_size,), maxval=num_classes, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensors((images, labels)).repeat()
    print(f"Compiling {model_name} (this wait is not an epoch)...")
    started = time.perf_counter()
    model.fit(dataset, epochs=1, steps_per_epoch=min(16, steps), verbose=0)
    print(f"Compile + warmup: {time.perf_counter() - started:.1f}s")
    timer = EpochTimer(steps, batch_size, limit_s=60.0)
    monitor = GpuMonitor()
    model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=0, callbacks=[timer, monitor])
    elapsed = timer.times[-1]
    projected = elapsed / steps * math.ceil(samples / batch_size)
    rate = (steps * batch_size) / elapsed
    print(
        f"Benchmark {model_name}: {elapsed:.1f}s for {steps} steps, {rate:,.0f} images/s. "
        f"Projected epoch on {samples:,} images: {projected:.1f}s."
    )
    print("This is the GPU compute ceiling. A real epoch also reads train.bin and runs the test set.")
    if projected <= 60:
        print("Projection is inside the 60 second target.")
    else:
        print("Projection is over 60 seconds. Use --model small_cnn, or prepare the data with --img-size 48.")
    return 0


def train_one(args, model_name: str) -> dict:
    warn_if_slow_mount(args.data_dir)
    meta = load_meta(args.data_dir)
    img_size = int(meta["img_size"])
    num_classes = int(meta["num_classes"])
    gpus = configure_runtime(not args.no_mixed_precision)
    describe_devices(gpus)
    print(DESCRIPTIONS[model_name])
    print(f"Output neurons: {num_classes} (one per character)")

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = choose_batch_size(model_name, img_size, num_classes)
        configure_runtime(not args.no_mixed_precision and bool(gpus))

    train_samples = int(meta["train_samples"])
    steps = train_samples // batch_size
    if steps < 1:
        raise SystemExit(
            f"Batch size {batch_size} is larger than the training set ({train_samples}). Lower --batch-size."
        )
    spe = max(1, min(args.steps_per_execution, steps))
    base_lr = args.lr if args.lr > 0 else min(4e-3, 1e-3 * math.sqrt(batch_size / 256.0))
    schedule = WarmupCosine(base_lr, total_steps=steps * args.epochs, warmup_steps=steps)

    model = build_model(model_name, img_size, num_classes)
    use_xla = (not args.no_xla) and bool(gpus)
    compile_model(model, schedule, num_classes, args.label_smoothing, spe, use_xla)

    run_dir = Path(args.runs) / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model_name,
        "description": DESCRIPTIONS[model_name],
        "img_size": img_size,
        "work_size": meta.get("work_size", img_size * 2),
        "num_classes": num_classes,
        "output_neurons": int(model.output_shape[-1]),
        "parameters": int(model.count_params()),
        "batch_size": batch_size,
        "epochs": args.epochs,
        "base_lr": base_lr,
        "label_smoothing": args.label_smoothing,
        "xla": use_xla,
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
        "steps_per_epoch": steps,
        "steps_per_execution": spe,
        "train_samples": train_samples,
        "test_samples": int(meta["test_samples"]),
        "data_dir": str(args.data_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "characters.json").write_text(
        json.dumps(meta["characters"], ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    print(f"Parameters: {model.count_params():,}. Batch {batch_size}. Steps/epoch {steps}. XLA {use_xla}.")

    train_ds = make_dataset(
        Path(args.data_dir) / meta["train_bin"],
        img_size,
        int(meta["record_bytes"]),
        batch_size,
        training=True,
        drop_remainder=True,
        shuffle_buffer=args.shuffle_buffer,
    )
    test_samples = int(meta["test_samples"])
    validation = None
    if test_samples >= batch_size:
        validation = make_dataset(
            Path(args.data_dir) / meta["test_bin"],
            img_size,
            int(meta["record_bytes"]),
            batch_size,
            training=False,
            drop_remainder=True,
            shuffle_buffer=0,
        )
    monitor_metric = "val_accuracy" if validation is not None else "accuracy"

    print("Compiling for this GPU. This wait is not counted as an epoch.")
    started = time.perf_counter()
    # Keep the warmup chunk equal to steps_per_execution so XLA compiles the same function fit() uses.
    warm_val_steps = None
    if validation is not None and (test_samples // batch_size) >= spe:
        warm_val_steps = spe
    model.fit(
        train_ds,
        epochs=1,
        steps_per_epoch=spe,
        validation_data=validation if warm_val_steps else None,
        validation_steps=warm_val_steps,
        verbose=0,
    )
    print(f"Compile + warmup: {time.perf_counter() - started:.1f}s")

    timer = EpochTimer(steps, batch_size, args.limit_seconds)
    callbacks = [
        timer,
        GpuMonitor(),
        NanStop(),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"),
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]
    if args.early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode="max",
                patience=args.early_stop,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps,
        validation_data=validation,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(run_dir / "last.keras")

    best_path = run_dir / "best.keras"
    if best_path.is_file():
        best = tf.keras.models.load_model(
            best_path,
            custom_objects={
                "SparseLabelSmoothingCE": SparseLabelSmoothingCE,
                "WarmupCosine": WarmupCosine,
            },
            compile=False,
        )
        weights = best.get_weights()
    else:
        weights = model.get_weights()
    eval_model = build_model(model_name, img_size, num_classes)
    eval_model.set_weights(weights)
    print("Scoring every test image...")
    metrics = evaluate_model(eval_model, args.data_dir, meta, batch_size)
    steady = _steady_seconds(timer.times)
    result = {
        **config,
        **metrics,
        "epoch_seconds": [round(item, 2) for item in timer.times],
        "steady_epoch_seconds": None if steady is None else round(steady, 2),
        "final_val_accuracy": float(history.history.get("val_accuracy", [None])[-1] or 0.0),
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"TEST RECOGNITION RATE ({model_name}): {metrics['recognition_rate_percent']:.2f}%  "
        f"top-5 {metrics['test_top5'] * 100:.2f}%"
    )
    if steady is not None:
        print(f"Steady epoch time (median from epoch 2, including the test pass): {steady:.1f}s")
    return result


def train_all(args) -> int:
    results = []
    for name in MODEL_NAMES:
        print(f"\n======== {name} ========")
        results.append(train_one(args, name))
        tf.keras.backend.clear_session()
    out = Path(args.runs) / "comparison.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    results.sort(key=lambda item: item["test_accuracy"], reverse=True)
    print("\nModel comparison (test recognition rate)")
    for item in results:
        print(
            f"  {item['model']:<12}  {item['recognition_rate_percent']:.2f}%  "
            f"top-5 {item['test_top5'] * 100:.2f}%  "
            f"epoch {item['steady_epoch_seconds']}s  "
            f"params {item['parameters']:,}"
        )
    print(f"Best model: {results[0]['model']}")
    print(f"Wrote {out}")
    return 0

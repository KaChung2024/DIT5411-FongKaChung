"""Warmup then cosine decay. Serializable so the saved model can reload the optimizer."""

from __future__ import annotations

import math

import tensorflow as tf

from dit5411.serialize import register_keras_serializable


@register_keras_serializable(package="dit5411")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(max(warmup_steps, 0))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        span = float(max(self.total_steps - self.warmup_steps, 1))
        progress = tf.clip_by_value((step - float(self.warmup_steps)) / span, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(progress * math.pi))
        decayed = self.base_lr * (0.01 + 0.99 * cosine)
        if self.warmup_steps <= 0:
            return decayed
        warm = self.base_lr * (step + 1.0) / float(self.warmup_steps)
        return tf.where(step < float(self.warmup_steps), warm, decayed)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

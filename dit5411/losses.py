"""Sparse label-smoothed cross-entropy that does not materialise a one-hot vector."""

from __future__ import annotations

import tensorflow as tf

from dit5411.serialize import register_keras_serializable


@register_keras_serializable(package="dit5411")
class SparseLabelSmoothingCE(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, smoothing: float = 0.05, name: str = "sparse_ls_ce"):
        super().__init__(name=name, reduction="sum_over_batch_size")
        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        if self.smoothing <= 0:
            return nll
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        uniform = -tf.reduce_mean(log_probs, axis=-1)
        keep = 1.0 - self.smoothing
        return keep * nll + self.smoothing * uniform

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "smoothing": self.smoothing})
        return config

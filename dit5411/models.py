"""Three classifiers. Every one ends in one logit per character (13,065 on the full set).

FastResNet is the model sized for an RTX 5080: wide enough that mixed-precision
Tensor Cores stay busy, small enough that 2,613,000 images of 64x64 stay under
a minute per epoch. Filter counts are multiples of 8 so Tensor Cores engage.
"""

from __future__ import annotations

import tensorflow as tf

MODEL_NAMES = ("mlp", "small_cnn", "fast_resnet")

DESCRIPTIONS = {
    "mlp": (
        "Fully connected baseline. The 64x64 image is flattened to 4,096 values, "
        "then Dense(1024) -> Dense(512) -> Dense(num_classes). No spatial filters, "
        "so it is the accuracy floor and the fastest epoch."
    ),
    "small_cnn": (
        "Six 3x3 convolutions with stride-2 downsampling, batch norm, global "
        "average pooling, then Dense(num_classes). A compact CNN between the MLP "
        "and the residual network."
    ),
    "fast_resnet": (
        "Residual CNN recommended for the RTX 5080. A stride-2 stem, then residual "
        "stages at 64, 128, 256 and 512 channels, global average pooling, and "
        "Dense(num_classes). About 0.5 billion multiply-adds per 64x64 image, "
        "which is the band that fills a 5080 and still finishes an epoch of "
        "2.61 million images inside 60 seconds."
    ),
}


def _logits(x, num_classes: int):
    name = f"logits_{num_classes}"
    # float32 logits are required under mixed_float16 so the softmax is stable.
    return tf.keras.layers.Dense(num_classes, dtype="float32", name=name)(x)


def _input(img_size: int):
    return tf.keras.Input(shape=(img_size, img_size, 1), dtype=tf.uint8, name="image")


def build_mlp(img_size: int, num_classes: int) -> tf.keras.Model:
    inp = _input(img_size)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale")(inp)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.Model(inp, _logits(x, num_classes), name="mlp")


def _conv_bn_relu(x, filters: int, stride: int):
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization(dtype="float32")(x)
    return tf.keras.layers.Activation("relu")(x)


def build_small_cnn(img_size: int, num_classes: int) -> tf.keras.Model:
    inp = _input(img_size)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale")(inp)
    x = _conv_bn_relu(x, 32, 2)
    x = _conv_bn_relu(x, 64, 1)
    x = _conv_bn_relu(x, 64, 2)
    x = _conv_bn_relu(x, 128, 1)
    x = _conv_bn_relu(x, 128, 2)
    x = _conv_bn_relu(x, 256, 2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.Model(inp, _logits(x, num_classes), name="small_cnn")


def _residual(x, filters: int, stride: int):
    shortcut = x
    x = _conv_bn_relu(x, filters, stride)
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization(dtype="float32")(x)
    in_channels = shortcut.shape[-1]
    if stride != 1 or in_channels != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(dtype="float32")(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation("relu")(x)


def build_fast_resnet(img_size: int, num_classes: int) -> tf.keras.Model:
    inp = _input(img_size)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale")(inp)
    # One cheap full-resolution layer, then leave 64x64 immediately. Two full
    # residual blocks at 64x64 would push the epoch past a minute.
    x = _conv_bn_relu(x, 32, 1)
    x = _conv_bn_relu(x, 64, 2)
    x = _residual(x, 64, 1)
    x = _residual(x, 128, 2)
    x = _residual(x, 128, 1)
    x = _residual(x, 256, 2)
    x = _residual(x, 256, 1)
    x = _residual(x, 512, 2)
    x = _residual(x, 512, 1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    return tf.keras.Model(inp, _logits(x, num_classes), name="fast_resnet")


BUILDERS = {
    "mlp": build_mlp,
    "small_cnn": build_small_cnn,
    "fast_resnet": build_fast_resnet,
}


def build_model(name: str, img_size: int, num_classes: int) -> tf.keras.Model:
    try:
        builder = BUILDERS[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown model {name!r}. Choose from {', '.join(MODEL_NAMES)}.") from exc
    model = builder(img_size, num_classes)
    if model.output_shape[-1] != num_classes:
        raise RuntimeError("output layer does not match the class count")
    return model

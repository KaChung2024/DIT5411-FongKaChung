"""Register a Keras class on TensorFlow builds that expose saving in different places."""

from __future__ import annotations


def register_keras_serializable(package: str = "dit5411"):
    def decorator(cls):
        register = _find_register()
        if register is None:
            return cls
        return register(package=package)(cls)

    return decorator


def _find_register():
    try:
        import keras

        register = getattr(getattr(keras, "saving", None), "register_keras_serializable", None)
        if register is not None:
            return register
    except Exception:
        pass
    try:
        import tensorflow as tf

        saving = getattr(tf.keras, "saving", None)
        register = getattr(saving, "register_keras_serializable", None)
        if register is not None:
            return register
        return getattr(tf.keras.utils, "register_keras_serializable", None)
    except Exception:
        return None

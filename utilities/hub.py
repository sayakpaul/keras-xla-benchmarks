"""
Utilities for handling TensorFlow Hub models.
"""

import tensorflow as tf
import tensorflow_hub as hub


def get_model_from_hub(url: str, input_resolution: int):
    """Initializes a tf.keras.Model from a TensorFlow Hub URL."""
    if "vit" not in url or "mixer" not in url:
        inputs = tf.keras.Input((input_resolution, input_resolution, 3))
        hub_module = hub.KerasLayer(url)
        outputs, _ = hub_module(inputs)
        return tf.keras.Model(inputs, outputs)
    else:
        return tf.keras.Sequential([hub.KerasLayer(url)])

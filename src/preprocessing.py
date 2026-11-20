"""Image preprocessing from the original crop training notebook."""

from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (128, 128)


def load_jpeg(path: str | Path) -> tf.Tensor:
    """Return an RGB image resized for the crop model, with pixels on the 0–255 scale."""
    return preprocess_jpeg(tf.io.read_file(str(path)))


def preprocess_jpeg(image_bytes: bytes | tf.Tensor) -> tf.Tensor:
    """Decode and resize JPEG bytes from a file or archive."""
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    return tf.image.resize(image, IMAGE_SIZE)

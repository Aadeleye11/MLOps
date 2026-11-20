"""Load the LeafVision model and check its serving signature."""

from pathlib import Path

import tensorflow as tf

from src import CROP_CLASSES
from src.preprocessing import IMAGE_SIZE


def load_crop_model(path: Path) -> tf.keras.Model:
    if not path.is_file():
        raise FileNotFoundError(f"Crop model file does not exist: {path}")
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if model.input_shape != (None, *IMAGE_SIZE, 3) or model.output_shape != (
        None, len(CROP_CLASSES)
    ):
        raise ValueError("Expected 128 × 128 RGB input and five crop scores")
    return model

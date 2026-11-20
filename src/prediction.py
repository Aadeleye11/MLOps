"""Classify a single JPEG with the verified crop model."""

from io import BytesIO

import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src import CROP_CLASSES
from src.preprocessing import preprocess_jpeg


MAX_IMAGE_PIXELS = 20_000_000


class InvalidImage(ValueError):
    """The input cannot be decoded as a JPEG."""


class ImageTooLarge(ValueError):
    """The decoded dimensions exceed the supported pixel count."""


def decode_jpeg_upload(data: bytes) -> tf.Tensor:
    try:
        with Image.open(BytesIO(data), formats=["JPEG"]) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageTooLarge(f"JPEG must have at most {MAX_IMAGE_PIXELS:,} pixels")
        return preprocess_jpeg(data)
    except Image.DecompressionBombError as error:
        raise ImageTooLarge("JPEG dimensions exceed the supported pixel count") from error
    except (UnidentifiedImageError, OSError, SyntaxError, tf.errors.InvalidArgumentError) as error:
        raise InvalidImage("Upload a readable JPEG image") from error


def predict_jpeg(model: tf.keras.Model, data: bytes) -> tuple[str, float]:
    pixels = decode_jpeg_upload(data)
    scores = model(pixels[tf.newaxis, ...], training=False)[0]
    tf.debugging.assert_all_finite(scores, "The crop model returned a non-finite score")
    tf.debugging.assert_near(tf.reduce_sum(scores), 1.0, atol=1e-5)
    index = int(tf.argmax(scores))
    return CROP_CLASSES[index], float(scores[index])

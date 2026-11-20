import tempfile
import unittest
from pathlib import Path

import tensorflow as tf

from src.preprocessing import load_jpeg


class LoadJpegTest(unittest.TestCase):
    def test_matches_notebook_pipeline_for_color_jpeg(self):
        pixels = tf.cast(tf.reshape(tf.range(24), (2, 4, 3)) * 11, tf.uint8)
        jpeg = tf.io.encode_jpeg(pixels, quality=95, chroma_downsampling=False)
        expected = tf.image.resize(tf.io.decode_jpeg(jpeg, channels=3), (128, 128))

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "leaf.jpg"
            tf.io.write_file(str(image_path), jpeg)
            actual = load_jpeg(image_path)

        self.assertEqual(actual.shape, (128, 128, 3))
        self.assertEqual(actual.dtype, tf.float32)
        tf.debugging.assert_equal(actual, expected)

    def test_decodes_grayscale_into_three_channels_without_normalizing(self):
        jpeg = tf.io.encode_jpeg(tf.fill((3, 5, 1), tf.constant(180, dtype=tf.uint8)))

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "gray_leaf.jpg"
            tf.io.write_file(str(image_path), jpeg)
            actual = load_jpeg(image_path)

        self.assertEqual(actual.shape, (128, 128, 3))
        tf.debugging.assert_near(actual, tf.fill((128, 128, 3), 180.0), atol=1.0)

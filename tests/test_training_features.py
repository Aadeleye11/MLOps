import unittest

from PIL import Image, ImageDraw

from scripts.explore_training_images import measure


class TrainingFeaturesTest(unittest.TestCase):
    def test_uniform_green_scores(self):
        features = measure(Image.new("RGB", (32, 64), (40, 140, 80)))
        self.assertAlmostEqual(features["brightness"], 103.26, places=3)
        self.assertAlmostEqual(features["green_dominance"], 60)
        self.assertEqual(features["edge_density"], 0)
        self.assertEqual(features["near_white"], 0)

    def test_high_contrast_boundary_counts_pixels_once(self):
        image = Image.new("RGB", (128, 128), "black")
        ImageDraw.Draw(image).rectangle((64, 0, 127, 127), fill="white")
        features = measure(image)
        self.assertAlmostEqual(features["edge_density"], 100 / 127)
        self.assertEqual(features["near_white"], 50)
        self.assertEqual(features["green_dominance"], 0)

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_splits import SPLITS, portable_path, prepare_splits


class PrepareSplitsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / "portable"
        for split in SPLITS:
            self.write_split(split, f"/content/work/rice/Original Dataset/{split}.jpg")

    def write_split(self, split, path, crop="rice"):
        with (self.root / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("path", "crop", "label"))
            writer.writerow((path, crop, "healthy"))

    def test_relative_paths_metadata_counts_and_rerun(self):
        for _ in range(2):
            counts = prepare_splits(self.root, self.output)
            for split in SPLITS:
                self.assertEqual(counts[split], {"rice": 1})
                with (self.output / f"{split}.csv").open(newline="") as stream:
                    self.assertEqual(list(csv.DictReader(stream)), [{
                        "path": f"rice/Original Dataset/{split}.jpg",
                        "crop": "rice",
                        "label": "healthy",
                    }])

    def test_cross_split_overlap_reports_both_locations_before_publish(self):
        self.write_split("test", "rice/Original Dataset/train.jpg")
        with self.assertRaisesRegex(ValueError, "train.csv line 2 and test.csv line 2"):
            prepare_splits(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_unknown_crop(self):
        self.write_split("val", "ccmt/leaf.jpg", crop="wheat")
        with self.assertRaisesRegex(ValueError, "unknown crop 'wheat'"):
            prepare_splits(self.root, self.output)

    def test_invalid_paths(self):
        for path in ("/tmp/rice/leaf.jpg", "/content/work/ccmt/../rice/a.jpg", "rice/"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                portable_path(path)

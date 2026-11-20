import tempfile
import unittest
from pathlib import Path

from src import CROP_CLASSES
from src.retraining import image_batches, select_examples


class RetrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.originals = [
            (f"{crop}-{split}", {"split": split, "path": f"{crop}-{split}.jpg", "crop": crop})
            for crop in CROP_CLASSES for split in ("train", "val", "test")
        ]
        self.uploads = [{"split": "upload", "crop": "rice", "path": "new.jpg", "sha256": "new"}]

    def test_selection_has_no_cross_split_hashes_or_repeats(self):
        originals = self.originals + [
            ("overlap", {"split": split, "path": f"{split}.jpg", "crop": "rice"})
            for split in ("train", "val")
        ] + [self.originals[1]]
        training, validation = select_examples(originals, self.uploads, per_crop=1)
        self.assertEqual({r["sha256"] for r in training}, {"new"} | {f"{c}-train" for c in CROP_CLASSES})
        self.assertEqual([r["sha256"] for r in validation], sorted(f"{c}-val" for c in CROP_CLASSES))
        self.assertEqual((training, validation), select_examples(originals, self.uploads, per_crop=1))

    def test_upload_matching_any_original_split_is_rejected(self):
        for split in ("train", "val", "test"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "original data"):
                select_examples(self.originals, [{**self.uploads[0], "sha256": f"rice-{split}"}], per_crop=1)

    def test_changed_upload_bytes_are_rejected_before_decoding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "new.jpg").write_bytes(b"different bytes")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                list(image_batches(self.uploads, {}, root))

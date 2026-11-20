import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.audit_image_hashes import ARCHIVES


class ImageHashAuditTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for source, filename in ARCHIVES.items():
            with zipfile.ZipFile(self.root / filename, "w") as archive:
                archive.writestr("a.jpg", b"same image bytes")
                archive.writestr("b.jpg", b"same image bytes")
                archive.writestr("c.jpg", source.encode())
        self.manifest("train", ["ccmt/a.jpg", "ccmt/b.jpg"])
        self.manifest("val", ["ccmt/c.jpg"])
        self.manifest("test", ["rice/c.jpg"])
        self.report = self.root / "report.json"

    def manifest(self, split, paths):
        with (self.root / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "crop"])
            writer.writerows((path, "rice" if path.startswith("rice/") else "maize")
                             for path in paths)

    def run_command(self, expected_status):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.audit_image_hashes",
             "--split-dir", self.root, "--archive-dir", self.root, "--report", self.report],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, expected_status, result.stderr)

    def test_within_and_cross_split_groups(self):
        self.run_command(0)
        self.manifest("test", ["rice/a.jpg"])
        self.run_command(1)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["images_hashed"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(report["unique_sha256"], 2)
        self.assertEqual(report["within_split_duplicate_groups"], {"train": 1, "val": 0, "test": 0})
        self.assertEqual(report["cross_split_duplicate_groups"], 1)
        group, = report["duplicate_groups"]
        self.assertEqual(group["sha256"], hashlib.sha256(b"same image bytes").hexdigest())
        self.assertEqual([row["split"] for row in group["members"]], ["train", "train", "test"])

    def test_missing_member_preserves_previous_report(self):
        self.run_command(0)
        original = self.report.read_bytes()
        self.manifest("val", ["rice/missing.jpg"])
        self.run_command(2)
        self.assertEqual(self.report.read_bytes(), original)

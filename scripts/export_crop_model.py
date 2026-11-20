"""Export the crop checkpoint to HDF5 and verify prediction parity."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_crop import test_batches
from src import CROP_CLASSES
from src.preprocessing import IMAGE_SIZE


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_predictions(original, exported, splits, archives):
    signature = ((None, *IMAGE_SIZE, 3), (None, len(CROP_CLASSES)))
    for model in (original, exported):
        if (model.input_shape, model.output_shape) != signature:
            raise ValueError("Expected 128 × 128 RGB input and five crop scores")

    support = Counter()
    correct = 0
    max_difference = 0.0
    for batch, images in test_batches(splits, archives):
        baseline = original(images, training=False).numpy()
        restored = exported(images, training=False).numpy()
        if not np.isfinite(baseline).all() or not np.isfinite(restored).all():
            raise ValueError("A model returned a non-finite score")
        if not np.allclose(baseline, restored, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Scores differ beyond tolerance in batch starting at {batch[0]['path']}")
        baseline_classes = baseline.argmax(axis=1)
        restored_classes = restored.argmax(axis=1)
        if not np.array_equal(baseline_classes, restored_classes):
            raise ValueError(f"Predicted crops differ in batch starting at {batch[0]['path']}")
        max_difference = max(max_difference, float(np.max(np.abs(baseline - restored))))
        for row, crop_index in zip(batch, restored_classes, strict=True):
            crop = row["crop"]
            support[crop] += 1
            correct += CROP_CLASSES[crop_index] == crop
    if set(support) != set(CROP_CLASSES):
        raise ValueError("Parity check requires labeled examples from all five crops")
    total = support.total()
    return {"images": total, "matching_classes": total, "correct": correct,
            "accuracy": correct / total, "support": dict(support),
            "max_absolute_score_difference": max_difference, "atol": 1e-7, "rtol": 1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/best_crop_mobilenetv2.keras"))
    parser.add_argument("--output", type=Path, default=Path("models/crop_model_v1.h5"))
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--archive-dir", type=Path, default=Path("artifacts/archives"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/crop-export-parity.json"))
    args = parser.parse_args()
    if args.output.suffix != ".h5" or args.output.resolve() == args.source.resolve():
        parser.error("--output must be a separate .h5 file")
    if args.report.suffix != ".json":
        parser.error("--report must be a .json file")

    original = tf.keras.models.load_model(args.source, compile=False, safe_mode=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / args.output.name
        original.save(staged, include_optimizer=False)
        exported = tf.keras.models.load_model(staged, compile=False, safe_mode=True)
        result = compare_predictions(original, exported, args.split_dir, args.archive_dir)
        result.update(source_sha256=checksum(args.source), h5_sha256=checksum(staged),
                      tensorflow=tf.__version__, keras=tf.keras.__version__,
                      class_order=CROP_CLASSES)
        staged.replace(args.output)
    content = json.dumps(result, indent=2, allow_nan=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()

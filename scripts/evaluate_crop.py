"""Evaluate the frozen crop checkpoint on the original and hash-filtered test sets."""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from contextlib import ExitStack
from itertools import batched
from pathlib import Path

import numpy as np
import tensorflow as tf

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_image_hashes import ARCHIVES, audit
from src import CROP_CLASSES
from src.evaluation import score, test_exclusions
from src.preprocessing import IMAGE_SIZE, preprocess_jpeg


def test_batches(splits: Path, archives: Path):
    with ExitStack() as stack:
        files = {key: stack.enter_context(zipfile.ZipFile(archives / name))
                 for key, name in ARCHIVES.items()}
        manifest = stack.enter_context((splits / "test.csv").open(encoding="utf-8", newline=""))
        for batch in batched(csv.DictReader(manifest), 32):
            images = []
            for row in batch:
                source, member = row["path"].split("/", 1)
                images.append(preprocess_jpeg(files[source].read(member)))
            yield batch, tf.stack(images)


def evaluate(model_path: Path, splits: Path, archives: Path) -> dict:
    with model_path.open("rb") as checkpoint:
        model_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    if model.input_shape != (None, *IMAGE_SIZE, 3) or model.output_shape != (None, len(CROP_CLASSES)):
        raise ValueError("Checkpoint shape does not match the five-class crop model")

    # Recompute membership from the supplied archives so a stale audit cannot select rows.
    exclusions = test_exclusions(audit(splits, archives)["duplicate_groups"])
    rows = []
    for batch, images in test_batches(splits, archives):
        probabilities = model(images, training=False).numpy()
        if not (np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
            raise ValueError("Checkpoint returned invalid class probabilities")
        for row, probs in zip(batch, probabilities, strict=True):
            actual = CROP_CLASSES.index(row["crop"])
            prediction = int(probs.argmax())
            rows.append({
                "path": row["path"], "crop": row["crop"],
                "prediction": CROP_CLASSES[prediction],
                "cross_entropy": float(-np.log(np.clip(float(probs[actual]), 1e-7, 1.0))),
                "exclusion": exclusions.get(row["path"]),
            })
    return {
        "checkpoint_sha256": model_hash,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "class_order": CROP_CLASSES,
        "original": score(rows),
        "hash_filtered": score([r for r in rows if r["exclusion"] is None]),
        "predictions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_crop_mobilenetv2.keras"))
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--archive-dir", type=Path, default=Path("artifacts/archives"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/crop-evaluation.json"))
    args = parser.parse_args()
    result = evaluate(args.model, args.split_dir, args.archive_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.report.parent) as temp:
        staged = Path(temp) / "report.json"
        staged.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(args.report)
    print(json.dumps({k: result[k] for k in ("original", "hash_filtered")}, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

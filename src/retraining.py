"""Fine-tune the saved crop classifier for one epoch on one labeled upload batch."""

import argparse
import hashlib
import json
import os
import random
import tempfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from itertools import batched
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import tensorflow as tf

from scripts.audit_image_hashes import ARCHIVES, hashed_images
from src import CROP_CLASSES
from src.evaluation import score
from src.model import load_crop_model
from src.preprocessing import preprocess_jpeg
from src.uploads import UploadStore

SEED = 42
PER_CROP = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def select_examples(hashed_rows, uploads, per_crop=PER_CROP):
    """Build a fixed validation set and balanced replay sample from split-exclusive hashes."""
    indexed = defaultdict(list)
    for digest, row in hashed_rows:
        indexed[digest].append(row)
    for row in uploads:
        if row["sha256"] in indexed:
            raise ValueError(f"Uploaded image already occurs in the original data: {row['sha256']}")

    by_crop = defaultdict(list)
    validation = []
    for digest, members in sorted(indexed.items()):
        splits = {m["split"] for m in members}
        if len(splits) > 1 or "test" in splits:
            continue
        row = members[0]
        if row["crop"] not in CROP_CLASSES or any(m["crop"] != row["crop"] for m in members):
            raise ValueError(f"Invalid or conflicting original crop label for {digest}")
        example = {**row, "sha256": digest}
        if row["split"] == "val":
            validation.append(example)
        elif row["split"] == "train":
            by_crop[row["crop"]].append(example)
    if {row["crop"] for row in validation} != set(CROP_CLASSES):
        raise ValueError("Validation requires examples of all five crops")
    rng = random.Random(SEED)
    training = list(uploads)
    for crop in CROP_CLASSES:
        if len(by_crop[crop]) < per_crop:
            raise ValueError(f"Need {per_crop} distinct training photos for {crop}")
        training.extend(rng.sample(by_crop[crop], per_crop))
    rng.shuffle(training)
    return training, validation


def image_batches(rows, archives, upload_root):
    for group in batched(rows, BATCH_SIZE):
        images = []
        for row in group:
            if row["split"] == "upload":
                data = (upload_root / row["path"]).read_bytes()
            else:
                source, member = row["path"].split("/", 1)
                data = archives[source].read(member)
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Image checksum mismatch: {row['path']}")
            images.append(preprocess_jpeg(data))
        yield group, tf.stack(images)


def evaluate_predictions(validation, probabilities):
    if not (np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Invalid validation probabilities")
    rows = []
    for example, values in zip(validation, probabilities, strict=True):
        actual = CROP_CLASSES.index(example["crop"])
        rows.append({"crop": example["crop"], "prediction": CROP_CLASSES[values.argmax()],
                     "cross_entropy": -np.log(np.clip(float(values[actual]), 1e-7, 1.0))})
    return score(rows)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retrain(store, batch_id, checkpoint, split_dir, archive_dir, output_dir,
            *, run_id=None, on_stage=None):
    batch = store.get_batch(batch_id)
    if not batch or not batch["images"]:
        raise ValueError(f"No stored images for batch {batch_id}")
    uploads = [{"split": "upload", "path": image["storage_path"], "sha256": image["sha256"],
                "crop": batch["crop"]} for image in batch["images"]]
    training, validation = select_examples(hashed_images(split_dir, archive_dir), uploads)
    tf.keras.utils.set_random_seed(SEED)
    original_hash = file_hash(checkpoint)
    baseline = load_crop_model(checkpoint)
    candidate = load_crop_model(checkpoint)
    head = candidate.layers[-1]
    if not isinstance(head, tf.keras.layers.Dense):
        raise ValueError("Expected a dense crop classification head")
    for layer in candidate.layers:
        layer.trainable = layer is head
    candidate.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss="sparse_categorical_crossentropy")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / str(run_id or uuid4())
    with tempfile.TemporaryDirectory(dir=output_dir) as temp, ExitStack() as stack:
        archives = {key: stack.enter_context(zipfile.ZipFile(archive_dir / name))
                    for key, name in ARCHIVES.items()}
        staged = Path(temp) / run_dir.name
        staged.mkdir()
        if on_stage:
            on_stage("training")
        for group, pixels in image_batches(training, archives, store.root):
            labels = np.array([CROP_CLASSES.index(row["crop"]) for row in group])
            candidate.train_on_batch(pixels, labels)
        weight_delta = float(np.max(np.abs(head.kernel.numpy() - baseline.layers[-1].kernel.numpy())))
        if not np.isfinite(weight_delta) or weight_delta == 0:
            raise ValueError("Training did not produce a finite change in the crop head weights")
        model_path = staged / "candidate.keras"
        candidate.save(model_path)
        restored = load_crop_model(model_path)

        if on_stage:
            on_stage("validating")
        baseline_scores, candidate_scores = [], []
        max_difference = 0.0
        for _, pixels in image_batches(validation, archives, store.root):
            before = candidate.predict_on_batch(pixels)
            after = restored.predict_on_batch(pixels)
            np.testing.assert_allclose(after, before, atol=1e-7, rtol=1e-6)
            np.testing.assert_array_equal(after.argmax(axis=1), before.argmax(axis=1))
            max_difference = max(max_difference, float(np.max(np.abs(after - before))))
            candidate_scores.append(after)
            baseline_scores.append(baseline.predict_on_batch(pixels))

        report = {
            "batch_id": str(batch_id), "checkpoint": str(checkpoint), "checkpoint_sha256": original_hash,
            "candidate": model_path.name, "candidate_sha256": file_hash(model_path),
            "class_order": CROP_CLASSES, "tensorflow": tf.__version__, "keras": tf.keras.__version__,
            "policy": {"epochs": 1, "trainable_layer": head.name, "optimizer": "Adam",
                       "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                       "original_images_per_crop": PER_CROP, "seed": SEED},
            "training_images": training, "validation_images": validation,
            "baseline": evaluate_predictions(validation, np.concatenate(baseline_scores)),
            "candidate_metrics": evaluate_predictions(validation, np.concatenate(candidate_scores)),
            "max_head_kernel_change": weight_delta, "max_reload_score_difference": max_difference,
        }
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.rename(run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True, type=UUID)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "best_crop_mobilenetv2.keras")
    parser.add_argument("--split-dir", type=Path, default=ARTIFACTS / "splits")
    parser.add_argument("--archive-dir", type=Path, default=ARTIFACTS / "archives")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "retraining")
    args = parser.parse_args()
    database = os.environ.get("LEAFVISION_DATABASE_URL")
    if not database:
        parser.error("Set LEAFVISION_DATABASE_URL to the labeled upload database")
    store = UploadStore(database, Path(os.environ.get("LEAFVISION_UPLOAD_ROOT", ARTIFACTS / "uploads")))
    print(retrain(store, args.batch_id, args.checkpoint, args.split_dir, args.archive_dir, args.output_dir))


if __name__ == "__main__":
    main()

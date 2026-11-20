"""Select and safely change the shared serving checkpoint."""

import math
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from src.model import load_crop_model
from src.retraining import ARTIFACTS, file_hash

DEFAULT_MODEL = ARTIFACTS.parent / "models" / "crop_model_v1.h5"
# Source and serving export verified by scripts/export_crop_model.py.
V1_SOURCE = "9bb1658122b2383ddb111dc35cfdb6b713e92954776ee5e67d0bff899f4b9d21"
V1_EXPORT = "876ce309d055c5d3ff310e21aadbd1964cecc020eddd9323d7b1cf5cc33bc07d"


class PromotionConflict(ValueError):
    """A version change cannot be applied to the current serving state."""


class UnknownJob(LookupError):
    """The requested training job does not exist."""


def baseline_path():
    return Path(os.environ.get("LEAFVISION_MODEL_PATH", DEFAULT_MODEL))


def initialize_registry(database, path):
    digest = file_hash(path)
    with psycopg.connect(database) as conn:
        conn.execute("""
            INSERT INTO leafvision_serving_model (baseline_version, baseline_sha256)
            VALUES (%s, %s) ON CONFLICT (singleton) DO NOTHING
        """, (path.stem, digest))
        baseline = conn.execute("SELECT baseline_version, baseline_sha256 FROM leafvision_serving_model").fetchone()
        if baseline != (path.stem, digest):
            raise ValueError("Configured baseline differs from the registered serving baseline")


def active_record(conn, *, lock=False):
    query = """
        SELECT r.active_job, r.baseline_version, r.baseline_sha256,
               COALESCE(r.active_job::text, r.baseline_version) AS version,
               COALESCE(j.result->>'candidate_sha256', r.baseline_sha256) AS sha256
        FROM leafvision_serving_model r
        LEFT JOIN leafvision_training_job j ON j.job_id = r.active_job
    """
    return conn.execute(query + (" FOR UPDATE OF r" if lock else "")).fetchone()


def artifact_path(record, baseline, directory):
    if record["active_job"] is None:
        return baseline
    return directory / str(record["active_job"]) / "candidate.keras"


def checked_hash(path, expected):
    if not path.is_file() or file_hash(path) != expected:
        raise PromotionConflict("Model artifact is missing or its SHA-256 differs from the registered value")


def training_checkpoint(database, directory):
    with psycopg.connect(database, row_factory=dict_row) as conn:
        record = active_record(conn)
    if record is None:  # A job queued before registry initialization.
        return ARTIFACTS / "best_crop_mobilenetv2.keras"
    path = artifact_path(record, baseline_path(), directory)
    digest = record["sha256"]
    if digest == V1_EXPORT:
        path, digest = ARTIFACTS / "best_crop_mobilenetv2.keras", V1_SOURCE
    checked_hash(path, digest)
    return path


def check_metrics(before, after):
    if (before["images"] <= 0 or before["images"] != after["images"]
            or before["support"] != after["support"]):
        raise PromotionConflict("Baseline and candidate validation support differs or is empty")
    for metrics in (before, after):
        for name in ("macro_f1", "accuracy", "mean_cross_entropy"):
            value = metrics[name]
            if not math.isfinite(value) or value < 0 or (name != "mean_cross_entropy" and value > 1):
                raise PromotionConflict("Invalid validation metrics")
    if after["macro_f1"] <= before["macro_f1"]:
        raise PromotionConflict("Validation macro F1 must improve")
    if after["accuracy"] < before["accuracy"]:
        raise PromotionConflict("Validation accuracy fell")
    if after["mean_cross_entropy"] > before["mean_cross_entropy"]:
        raise PromotionConflict("Validation loss increased")


def change_version(database, baseline, directory, expected, job_id=None):
    """Lock the active pointer, validate the destination, then commit a single switch."""
    with psycopg.connect(database, row_factory=dict_row) as conn:
        current = active_record(conn, lock=True)
        if current["version"] != expected:
            raise PromotionConflict(f"Active version is {current['version']}")
        if job_id is None:
            version, path, digest = current["baseline_version"], baseline, current["baseline_sha256"]
        else:
            job = conn.execute("SELECT status, result FROM leafvision_training_job WHERE job_id = %s",
                               (job_id,)).fetchone()
            if job is None:
                raise UnknownJob("Training job not found")
            if job["status"] != "succeeded":
                raise PromotionConflict("Training job must have succeeded")
            result = job["result"]
            expected_source = V1_SOURCE if current["sha256"] == V1_EXPORT else current["sha256"]
            if result.get("checkpoint_sha256") != expected_source:
                raise PromotionConflict("The comparison checkpoint differs from the active model")
            try:
                check_metrics(result["baseline"], result["candidate_metrics"])
                digest = result["candidate_sha256"]
            except (KeyError, TypeError) as exc:
                raise PromotionConflict("Completed job lacks valid comparison metrics or checksum") from exc
            version, path = str(job_id), directory / str(job_id) / "candidate.keras"
        checked_hash(path, digest)
        load_crop_model(path)
        conn.execute("UPDATE leafvision_serving_model SET active_job = %s", (job_id,))
        return {"model_version": version, "sha256": digest}

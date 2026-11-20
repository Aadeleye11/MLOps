"""Consume durable retraining jobs, one at a time, in a separate process."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.retraining import ARTIFACTS, file_hash, retrain
from src.registry import training_checkpoint
from src.uploads import UploadStore


def train_job(store, job, output_dir, on_stage):
    job_id, batch_id = job["job_id"], job["batch_id"]
    run_dir = output_dir / str(job_id)
    # A crash after publication but before the status commit leaves a complete run.
    if not run_dir.exists():
        retrain(store, batch_id, training_checkpoint(store.database_url, output_dir),
                ARTIFACTS / "splits", ARTIFACTS / "archives", output_dir,
                run_id=job_id, on_stage=on_stage)
    report = json.loads((run_dir / "report.json").read_text())
    if report["batch_id"] != str(batch_id) or file_hash(run_dir / "candidate.keras") != report["candidate_sha256"]:
        raise ValueError("The saved candidate does not match the job report")
    return {"candidate_version": str(job_id), **{key: report[key] for key in (
        "checkpoint_sha256", "candidate_sha256", "baseline", "candidate_metrics", "max_reload_score_difference"
    )}}


def process_one(store, output_dir):
    # Session lock spans training. Autocommit makes each status visible to other sessions.
    with psycopg.connect(store.database_url, autocommit=True, row_factory=dict_row) as connection:
        if not connection.execute("SELECT pg_try_advisory_lock(72545109) AS acquired").fetchone()["acquired"]:
            return False
        job = connection.execute("""
            UPDATE leafvision_training_job
            SET status = 'running', stage = 'preparing_data',
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP)
            WHERE job_id = (
                SELECT job_id FROM leafvision_training_job WHERE status IN ('queued', 'running')
                ORDER BY (status = 'running') DESC, created_at, job_id LIMIT 1
            ) RETURNING job_id, batch_id
        """).fetchone()
        if job is None:
            return False
        job_id = job["job_id"]

        def on_stage(stage):
            connection.execute("UPDATE leafvision_training_job SET stage = %s WHERE job_id = %s",
                               (stage, job_id))

        try:
            result = train_job(store, job, output_dir, on_stage)
        except Exception as exc:
            logging.exception("Retraining job %s failed", job_id)
            message = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: see worker log"
            connection.execute("""
                UPDATE leafvision_training_job SET status = 'failed', stage = NULL,
                    error = %s, finished_at = CURRENT_TIMESTAMP WHERE job_id = %s
            """, (message, job_id))
        else:
            connection.execute("""
                UPDATE leafvision_training_job SET status = 'succeeded', stage = NULL,
                    result = %s, finished_at = CURRENT_TIMESTAMP WHERE job_id = %s
            """, (Jsonb(result), job_id))
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Process at most one pending job and exit")
    args = parser.parse_args()
    database = os.environ.get("LEAFVISION_DATABASE_URL")
    if not database:
        parser.error("Set LEAFVISION_DATABASE_URL to the API's upload database")
    store = UploadStore(database, Path(os.environ.get("LEAFVISION_UPLOAD_ROOT", ARTIFACTS / "uploads")))
    store.initialize()
    while True:
        processed = process_one(store, ARTIFACTS / "retraining")
        if args.once:
            return
        if not processed:
            time.sleep(1)


if __name__ == "__main__":
    main()

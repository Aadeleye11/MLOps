"""Persist retraining requests and expose their status to API callers."""

from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row


def enqueue(database_url: str, batch_id: UUID) -> dict | None:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return connection.execute("""
            INSERT INTO leafvision_training_job (job_id, batch_id)
            SELECT %s, id FROM leafvision_upload_batch WHERE id = %s
            ON CONFLICT (batch_id) WHERE status IN ('queued', 'running')
            DO UPDATE SET batch_id = EXCLUDED.batch_id
            RETURNING job_id, batch_id, status
        """, (uuid4(), batch_id)).fetchone()


def get_job(database_url: str, job_id: UUID) -> dict | None:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return connection.execute("""
            SELECT job_id, batch_id, status, stage, created_at, started_at, finished_at, result, error
            FROM leafvision_training_job WHERE job_id = %s
        """, (job_id,)).fetchone()

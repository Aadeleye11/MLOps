"""PostgreSQL metadata and filesystem storage for labeled JPEG batches."""

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from src import CROP_CLASSES


class DuplicateImage(ValueError):
    """The same image bytes occur twice or were already stored."""


@dataclass(frozen=True)
class UploadStore:
    database_url: str
    root: Path

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with psycopg.connect(self.database_url) as connection:
            connection.execute("SELECT pg_advisory_xact_lock(72545108)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS leafvision_upload_batch (
                    id UUID PRIMARY KEY,
                    crop TEXT NOT NULL CHECK (crop IN ('cashew', 'cassava', 'maize', 'rice', 'tomato')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS leafvision_uploaded_image (
                    sha256 TEXT PRIMARY KEY,
                    batch_id UUID NOT NULL REFERENCES leafvision_upload_batch (id),
                    storage_path TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0)
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS leafvision_uploaded_image_batch_idx
                ON leafvision_uploaded_image (batch_id)
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS leafvision_training_job (
                    job_id UUID PRIMARY KEY,
                    batch_id UUID NOT NULL REFERENCES leafvision_upload_batch (id),
                    status TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
                    stage TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT
                )
            """)
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS leafvision_active_batch_job
                ON leafvision_training_job (batch_id) WHERE status IN ('queued', 'running')
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS leafvision_serving_model (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                    baseline_version TEXT NOT NULL,
                    baseline_sha256 TEXT NOT NULL,
                    active_job UUID REFERENCES leafvision_training_job(job_id)
                )
            """)

    def save_batch(self, crop: str, images: list[bytes]) -> UUID:
        if crop not in CROP_CLASSES or not images:
            raise ValueError("A batch requires a supported crop and at least one image")
        batch_id = uuid4()
        final = self.root / str(batch_id)
        rows = []
        with tempfile.TemporaryDirectory(prefix=".batch-", dir=self.root) as temp:
            staged = Path(temp) / str(batch_id)
            staged.mkdir()
            seen = set()
            for data in images:
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    raise DuplicateImage("The batch contains duplicate image bytes")
                seen.add(digest)
                filename = f"{digest}.jpg"
                with (staged / filename).open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append((digest, batch_id, f"{batch_id}/{filename}", len(data)))
            sync_directory(staged)

            with psycopg.connect(self.database_url) as connection:
                connection.execute(
                    "INSERT INTO leafvision_upload_batch (id, crop) VALUES (%s, %s)", (batch_id, crop)
                )
                try:
                    with connection.cursor() as cursor:
                        cursor.executemany("""
                            INSERT INTO leafvision_uploaded_image
                            (sha256, batch_id, storage_path, size_bytes) VALUES (%s, %s, %s, %s)
                        """, rows)
                except psycopg.errors.UniqueViolation as error:
                    raise DuplicateImage("An image with these bytes is already stored") from error
                staged.rename(final)
                try:
                    sync_directory(self.root)
                except OSError:
                    shutil.rmtree(final)
                    raise
                # Leave the directory in place if the commit outcome is ambiguous.
                # Removing it could break records committed before a connection loss.
        return batch_id

    def get_batch(self, batch_id: UUID) -> dict | None:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            batch = connection.execute(
                "SELECT id, crop, created_at FROM leafvision_upload_batch WHERE id = %s", (batch_id,)
            ).fetchone()
            if batch is None:
                return None
            images = connection.execute("""
                SELECT sha256, storage_path, size_bytes FROM leafvision_uploaded_image
                WHERE batch_id = %s ORDER BY sha256
            """, (batch_id,)).fetchall()
        return {**batch, "images": images}

    def list_batches(self) -> list[dict]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            return connection.execute("""
                SELECT b.id AS batch_id, b.crop, b.created_at, count(i.sha256) AS image_count,
                       (SELECT j.job_id FROM leafvision_training_job j WHERE j.batch_id = b.id
                        ORDER BY j.created_at DESC, j.job_id DESC LIMIT 1) AS latest_job_id
                FROM leafvision_upload_batch b
                JOIN leafvision_uploaded_image i ON i.batch_id = b.id
                GROUP BY b.id ORDER BY b.created_at DESC, b.id DESC
            """).fetchall()

    def summary(self) -> dict:
        with psycopg.connect(self.database_url) as connection:
            counts = connection.execute("""
                SELECT b.crop, count(DISTINCT b.id), count(i.sha256)
                FROM leafvision_upload_batch AS b
                LEFT JOIN leafvision_uploaded_image AS i ON i.batch_id = b.id
                GROUP BY b.crop
            """).fetchall()
        images_by_crop = dict.fromkeys(CROP_CLASSES, 0)
        total_batches = 0
        for crop, batches, images in counts:
            images_by_crop[crop] = images
            total_batches += batches
        return {
            "total_batches": total_batches,
            "total_images": sum(images_by_crop.values()),
            "images_by_crop": images_by_crop,
        }


def sync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

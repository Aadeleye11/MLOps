"""Data returned by the LeafVision API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Prediction(BaseModel):
    crop: str
    score: float
    model_version: str
    latency_ms: float


class UploadReceipt(BaseModel):
    batch_id: UUID
    crop: str
    image_count: int


class UploadSummary(BaseModel):
    total_batches: int
    total_images: int
    images_by_crop: dict[str, int]


class PredictionFailure(BaseModel):
    time: datetime
    status: int


class MonitoringSummary(BaseModel):
    uptime_seconds: float
    model_version: str | None
    requests: int
    failures: int
    latency_samples: int
    latency_window: int
    mean_ms: float | None
    p95_ms: float | None
    recent_failures: list[PredictionFailure]

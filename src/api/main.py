"""Local API for the verified LeafVision crop classifier."""

import os
from contextlib import asynccontextmanager
from hmac import compare_digest
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src import CROP_CLASSES
from src.api.monitoring import PredictionMetrics, PredictionTimingMiddleware
from src.jobs import enqueue, get_job
from src.model import load_crop_model
from src.prediction import ImageTooLarge, InvalidImage, decode_jpeg_upload, predict_jpeg
from src.preprocessing import IMAGE_SIZE
from src.schemas import MonitoringSummary, Prediction, UploadReceipt, UploadSummary
from src.registry import (ARTIFACTS, UnknownJob, active_record, artifact_path, baseline_path,
                          change_version, checked_hash, initialize_registry)
from src.uploads import DuplicateImage, UploadStore


MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_BATCH_FILES = 10
DEFAULT_UPLOAD_ROOT = ARTIFACTS / "uploads"


class RetrainingRequest(BaseModel):
    batch_id: UUID


class VersionChange(BaseModel):
    expected_active: str


@asynccontextmanager
async def lifespan(application: FastAPI):
    path = baseline_path()
    database_url = os.environ.get("LEAFVISION_DATABASE_URL")
    store = None
    if database_url:
        root = Path(os.environ.get("LEAFVISION_UPLOAD_ROOT", str(DEFAULT_UPLOAD_ROOT)))
        store = UploadStore(database_url, root)
        store.initialize()
        initialize_registry(database_url, path)
    application.state.upload_store = store
    application.state.baseline = path
    application.state.model_lock = Lock()
    application.state.model = None
    try:
        if store is None:
            application.state.model = (path.stem, load_crop_model(path))
        else:
            serving_model(application.state)
        application.state.ready_since = monotonic()
        application.state.prediction_metrics = PredictionMetrics()
        yield
    finally:
        if hasattr(application.state, "prediction_metrics"):
            del application.state.prediction_metrics
        del application.state.model
        del application.state.upload_store


app = FastAPI(title="LeafVision Crop API", lifespan=lifespan)
app.add_middleware(PredictionTimingMiddleware)


def require_operator(credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False))):
    expected = os.environ.get("LEAFVISION_OPERATOR_TOKEN")
    if not expected or not expected.strip():
        raise HTTPException(status_code=503, detail="Operator access is not configured")
    if credentials is None or not compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Enter a valid operator token",
                            headers={"WWW-Authenticate": "Bearer"})


operator = APIRouter(dependencies=[Depends(require_operator)])


@operator.get("/operator/access", status_code=204)
def operator_access():
    """Let the dashboard verify the operator before enabling controls."""
    return Response(status_code=204)


def ready_state(request: Request):
    state = request.app.state
    if not hasattr(state, "model"):
        raise HTTPException(status_code=503, detail="Crop model is not ready")
    return state


def serving_model(state):
    if state.upload_store is None:
        return state.model
    with state.model_lock:
        try:
            with psycopg.connect(state.upload_store.database_url, row_factory=dict_row) as conn:
                active = active_record(conn)
            if state.model is None or state.model[0] != active["version"]:
                path = artifact_path(active, state.baseline, ARTIFACTS / "retraining")
                checked_hash(path, active["sha256"])
                state.model = (active["version"], load_crop_model(path))
        except (psycopg.Error, ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail="Active model is unavailable") from exc
        return state.model


@app.get("/health")
def health(request: Request):
    state = ready_state(request)
    serving_model(state)
    return {"status": "ready", "uptime_seconds": round(monotonic() - state.ready_since, 3)}


@app.get("/model/info")
def model_info(request: Request):
    state = ready_state(request)
    version, _ = serving_model(state)
    return {"model_version": version, "baseline_version": state.baseline.stem,
            "classes": CROP_CLASSES, "input_shape": (*IMAGE_SIZE, 3)}


@app.get("/monitoring/summary")
def monitoring_summary(request: Request) -> MonitoringSummary:
    state = ready_state(request)
    try:
        version, _ = serving_model(state)
    except HTTPException:
        version = None
    return {
        "uptime_seconds": round(monotonic() - state.ready_since, 3),
        "model_version": version,
        **state.prediction_metrics.snapshot(),
    }


@app.post("/predict")
def predict(request: Request, file: UploadFile) -> Prediction:
    state = ready_state(request)
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="JPEG must be 5 MiB or smaller")

    start = perf_counter()
    version, model = serving_model(state)
    try:
        crop, score = predict_jpeg(model, data)
    except InvalidImage as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ImageTooLarge as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    return Prediction(crop=crop, score=score, model_version=version,
                      latency_ms=round((perf_counter() - start) * 1000, 3))


def upload_store(request: Request) -> UploadStore:
    store = ready_state(request).upload_store
    if store is None:
        raise HTTPException(status_code=503, detail="Labeled upload storage is not configured")
    return store


@operator.post("/data/upload", status_code=201)
def upload_batch(request: Request, files: list[UploadFile], crop: str = Form()) -> UploadReceipt:
    store = upload_store(request)
    if crop not in CROP_CLASSES:
        raise HTTPException(status_code=422, detail=f"Crop must be one of {', '.join(CROP_CLASSES)}")
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(status_code=422, detail="Upload between 1 and 10 JPEGs")
    images = []
    for index, file in enumerate(files, start=1):
        data = file.file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"File {index} must be 5 MiB or smaller")
        try:
            decode_jpeg_upload(data)
        except InvalidImage as error:
            raise HTTPException(status_code=422, detail=f"File {index}: {error}") from error
        except ImageTooLarge as error:
            raise HTTPException(status_code=413, detail=f"File {index}: {error}") from error
        images.append(data)

    try:
        batch_id = store.save_batch(crop, images)
    except DuplicateImage as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"batch_id": batch_id, "crop": crop, "image_count": len(images)}


@app.get("/data/summary")
def upload_summary(request: Request) -> UploadSummary:
    """Count accepted upload batches and images for each crop."""
    return upload_store(request).summary()


@app.get("/data/batches/{batch_id}")
def get_batch(request: Request, batch_id: UUID):
    batch = upload_store(request).get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Upload batch not found")
    return batch


@app.get("/data/batches")
def list_batches(request: Request):
    return upload_store(request).list_batches()


@operator.post("/train/retrain", status_code=202)
def request_retraining(request: Request, body: RetrainingRequest, response: Response):
    job = enqueue(upload_store(request).database_url, body.batch_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Upload batch not found")
    response.headers["Location"] = f"/train/jobs/{job['job_id']}"
    return job


@app.get("/train/jobs/{job_id}")
def training_status(request: Request, job_id: UUID):
    job = get_job(upload_store(request).database_url, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


def switch_model(request, body, job_id=None):
    store = upload_store(request)
    try:
        return change_version(store.database_url, request.app.state.baseline, ARTIFACTS / "retraining",
                              body.expected_active, job_id)
    except UnknownJob as error:
        raise HTTPException(status_code=404, detail="Training job not found") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@operator.post("/model/versions/{job_id}/promote")
def promote_model(request: Request, job_id: UUID, body: VersionChange):
    return switch_model(request, body, job_id)


@operator.post("/model/rollback")
def rollback_model(request: Request, body: VersionChange):
    return switch_model(request, body)


app.include_router(operator)

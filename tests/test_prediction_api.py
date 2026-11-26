import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from src.api.main import MAX_FILE_BYTES, app, lifespan
from src.api.monitoring import PredictionMetrics
from src.model import load_crop_model
from src.prediction import predict_jpeg


class PredictionAPITest(unittest.IsolatedAsyncioTestCase):
    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def test_prediction_matches_offline_and_loads_model_once(self):
        image = io.BytesIO()
        Image.new("RGB", (48, 32), "green").save(image, format="JPEG")
        data = image.getvalue()
        path = Path("models/crop_model_v1.h5")
        crop, score = predict_jpeg(load_crop_model(path), data)
        with (
            patch.dict(os.environ, {"LEAFVISION_MODEL_PATH": str(path)}),
            patch("src.api.main.load_crop_model", wraps=load_crop_model) as load,
        ):
            async with lifespan(app), self.client() as client:
                self.assertEqual((await client.get("/health")).json()["status"], "ready")
                self.assertEqual((await client.get("/model/info")).json()["classes"],
                                 ["cashew", "cassava", "maize", "rice", "tomato"])
                response = await client.post("/predict", files={"file": ("leaf.jpg", data)})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["crop"], crop)
                self.assertAlmostEqual(response.json()["score"], score, places=7)
                self.assertEqual(response.json()["model_version"], "crop_model_v1")
            load.assert_called_once_with(path)

    async def test_operator_actions_require_a_configured_matching_bearer_token(self):
        job_id = "00000000-0000-4000-8000-000000000001"
        change = {"json": {"expected_active": "crop_model_v1"}}
        routes = [
            ("GET", "/operator/access", {}),
            ("POST", "/data/upload", {"files": {"files": ("leaf.jpg", b"test")}, "data": {"crop": "rice"}}),
            ("POST", "/train/retrain", {"json": {"batch_id": job_id}}),
            ("POST", f"/model/versions/{job_id}/promote", change),
            ("POST", "/model/rollback", change),
        ]
        async with self.client() as client:
            for method, path, body in routes:
                with self.subTest(path=path), patch.dict(os.environ, LEAFVISION_OPERATOR_TOKEN=""):
                    result = await client.request(method, path, headers={"Authorization": "Bearer valid"}, **body)
                    self.assertEqual(result.status_code, 503)
                    self.assertEqual(result.json()["detail"], "Operator access is not configured")
                with self.subTest(path=path), patch.dict(os.environ, LEAFVISION_OPERATOR_TOKEN="valid"):
                    for header in ("", "Basic valid", "Bearer incorrect"):
                        result = await client.request(method, path, headers={"Authorization": header}, **body)
                        self.assertEqual(result.status_code, 401)
                        self.assertEqual(result.headers["WWW-Authenticate"], "Bearer")
            with patch.dict(os.environ, LEAFVISION_OPERATOR_TOKEN="valid"):
                result = await client.get("/operator/access", headers={"Authorization": "Bearer valid"})
                self.assertEqual(result.status_code, 204)

    async def test_invalid_uploads(self):
        png = io.BytesIO()
        Image.new("RGB", (3, 3)).save(png, format="PNG")
        with patch("src.api.main.load_crop_model"):
            async with lifespan(app), self.client() as client:
                self.assertEqual((await client.post("/predict")).status_code, 422)
                for data in (b"not an image", png.getvalue()):
                    with self.subTest(data=data[:8]):
                        response = await client.post("/predict", files={"file": ("leaf.jpg", data)})
                        self.assertEqual(response.status_code, 422)
                        self.assertIn("readable JPEG", response.json()["detail"])
                large = {"file": ("large.jpg", b"0" * (MAX_FILE_BYTES + 1))}
                self.assertEqual((await client.post("/predict", files=large)).status_code, 413)

    async def test_summary_requires_configured_upload_storage(self):
        with patch.dict(os.environ, {"LEAFVISION_DATABASE_URL": ""}), patch("src.api.main.load_crop_model"):
            async with lifespan(app), self.client() as client:
                self.assertEqual((await client.get("/data/summary")).status_code, 503)

    async def test_monitoring_counts_validation_and_server_errors_and_resets(self):
        with patch.dict(os.environ, {"LEAFVISION_DATABASE_URL": ""}), patch("src.api.main.load_crop_model"):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with lifespan(app), httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                empty = (await client.get("/monitoring/summary")).json()
                self.assertEqual(empty["requests"], 0)
                self.assertIsNone(empty["p95_ms"])
                self.assertEqual((await client.post("/predict")).status_code, 422)
                with patch("src.api.main.predict_jpeg", return_value=("rice", .9)):
                    self.assertEqual((await client.post("/predict", files={"file": ("a.jpg", b"a")})).status_code, 200)
                with patch("src.api.main.predict_jpeg", side_effect=RuntimeError("unavailable")):
                    self.assertEqual((await client.post("/predict", files={"file": ("b.jpg", b"b")})).status_code, 500)
                await client.get("/health")
                stats = (await client.get("/monitoring/summary")).json()
                self.assertEqual((stats["requests"], stats["failures"], stats["latency_samples"]), (3, 2, 1))
                self.assertGreater(stats["mean_ms"], 0)
                self.assertEqual(stats["mean_ms"], stats["p95_ms"])
                self.assertEqual([f["status"] for f in stats["recent_failures"]], [500, 422])
            async with lifespan(app), self.client() as client:
                self.assertEqual((await client.get("/monitoring/summary")).json()["requests"], 0)

    def test_latency_window_and_failure_history_are_bounded(self):
        metrics = PredictionMetrics()
        for elapsed_ms in range(1, 102):
            metrics.record(200, elapsed_ms)
        for _ in range(12):
            metrics.record(422, 1)
        result = metrics.snapshot()
        self.assertEqual((result["requests"], result["failures"], result["latency_samples"]), (113, 12, 100))
        self.assertEqual((result["mean_ms"], result["p95_ms"]), (51.5, 96))
        self.assertEqual(len(result["recent_failures"]), 10)

    async def test_startup_fails_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.h5"
            with patch.dict(os.environ, {"LEAFVISION_MODEL_PATH": str(missing)}):
                with self.assertRaisesRegex(FileNotFoundError, "Crop model file does not exist"):
                    async with lifespan(app):
                        self.fail("Startup should fail for a missing model")
        async with self.client() as client:
            self.assertEqual((await client.get("/health")).status_code, 503)

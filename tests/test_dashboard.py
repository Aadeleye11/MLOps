import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pyarrow as pa
from PIL import Image
from streamlit.testing.v1 import AppTest


class DashboardTest(unittest.TestCase):
    def test_insights_charts_show_training_summary_and_link_back_to_prediction(self):
        frontend = Path(__file__).resolve().parents[1] / "frontend"
        summary = json.loads((frontend / "data/training-summary.json").read_text())
        app = AppTest.from_file(str(frontend / "app.py")).run()
        app.switch_page("dataset_insights.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Dataset insights")
        charts = app.get("vega_lite_chart")
        for chart, feature in zip(charts, ("brightness", "green_dominance", "edge_density"), strict=True):
            rows = pa.ipc.open_stream(chart.proto.data.data).read_all().to_pylist()
            expected = [{"crop": crop.title(), "images": stats["images"], **stats[feature]}
                        for crop, stats in summary.items()]
            self.assertEqual(rows, expected)
        app.switch_page("crop_prediction.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.file_uploader[0].label, "Leaf photo")

    def test_result_belongs_to_selected_photo_and_failed_retry_clears_it(self):
        app = AppTest.from_file("../frontend/app.py").run()
        self.assertTrue(app.button[0].disabled)
        jpeg = io.BytesIO()
        Image.new("RGB", (16, 16), "green").save(jpeg, format="JPEG")
        app.file_uploader[0].set_value(("leaf.jpg", jpeg.getvalue(), "image/jpeg")).run()
        request = httpx.Request("POST", "http://test/predict")
        success = httpx.Response(200, request=request, json={
            "crop": "maize", "score": 0.875, "model_version": "candidate-2", "latency_ms": 5.0,
        })
        with patch("httpx.post", return_value=success) as post:
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(post.call_args.kwargs["files"]["file"], ("leaf.jpg", jpeg.getvalue(), "image/jpeg"))
            self.assertEqual([m.value for m in app.metric], ["Maize", "0.87500000"])
            self.assertEqual(app.text[0].value, "candidate-2")
            app.run()
            post.assert_called_once()
            self.assertEqual(app.text[0].value, "candidate-2")
            post.side_effect = httpx.ConnectError("unavailable")
            app.button[0].click().run()
            self.assertTrue(app.error)
            self.assertFalse(app.metric)
            post.side_effect = None
            app.button[0].click().run()
            app.file_uploader[0].set_value(("other.jpg", b"bad", "image/jpeg")).run()
            self.assertFalse(app.metric)
            post.return_value = httpx.Response(422, request=request, json={"detail": "Upload a readable JPEG image"})
            app.button[0].click().run()
            self.assertEqual(app.error[0].value, "Upload a readable JPEG image")

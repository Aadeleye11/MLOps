import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
import psycopg
from fastapi.testclient import TestClient
from PIL import Image
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from streamlit.testing.v1 import AppTest

from src.api.main import app, lifespan
from src.worker import process_one
from src.registry import V1_SOURCE, training_checkpoint


TEST_TOKEN = "test-operator-token"
OPERATOR_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}


def jpeg(color):
    output = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(output, format="JPEG")
    return output.getvalue()


@unittest.skipUnless(os.environ.get("LEAFVISION_TEST_DATABASE_URL"), "requires a PostgreSQL test database")
class UploadAPITest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        database = os.environ["LEAFVISION_TEST_DATABASE_URL"]
        schema_name = f"leafvision_test_{uuid4().hex}"
        schema = sql.Identifier(schema_name)
        with psycopg.connect(database) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        self.addCleanup(self.drop_schema, database, schema)
        self.database = make_conninfo(database, options=f"-c search_path={schema_name}")
        uploads = tempfile.TemporaryDirectory()
        self.addCleanup(uploads.cleanup)
        self.root = Path(uploads.name)
        env = patch.dict(os.environ, {
            "LEAFVISION_DATABASE_URL": self.database,
            "LEAFVISION_OPERATOR_TOKEN": TEST_TOKEN,
            "LEAFVISION_UPLOAD_ROOT": str(self.root),
        })
        self.addCleanup(env.stop)
        env.start()
        model = patch("src.api.main.load_crop_model")
        self.addCleanup(model.stop)
        model.start()

    @staticmethod
    def drop_schema(database, schema):
        with psycopg.connect(database) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                 headers=OPERATOR_HEADERS)

    async def upload(self, client, *images, crop="rice"):
        files = [("files", (f"leaf-{index}.jpg", data, "image/jpeg"))
                 for index, data in enumerate(images)]
        return await client.post("/data/upload", data={"crop": crop}, files=files)

    def test_dashboard_saves_batch_refreshes_counts_and_handles_duplicate(self):
        ui = AppTest.from_file("../frontend/app.py", default_timeout=30).run()
        images = [jpeg("green"), jpeg("yellow")]
        with TestClient(app) as api, patch("httpx.get", side_effect=api.get), patch("httpx.post", side_effect=api.post) as post:
            ui.switch_page("bulk_upload.py").run()
            self.assertFalse(ui.exception)
            self.assertTrue(ui.button[0].disabled)
            self.assertTrue(ui.file_uploader[0].disabled)
            ui.text_input(key="operator_token").set_value("incorrect").run()
            self.assertTrue(ui.button[0].disabled)
            self.assertTrue(ui.file_uploader[0].disabled)
            ui.text_input(key="operator_token").set_value(TEST_TOKEN).run()
            self.assertFalse(ui.file_uploader[0].disabled)
            ui.selectbox[0].select("maize")
            ui.file_uploader[0].set_value([(f"leaf-{i}.jpg", data, "image/jpeg")
                                           for i, data in enumerate(images)]).run()
            self.assertFalse(ui.button[0].disabled)
            ui.button[0].click().run()
            self.assertFalse(ui.exception)
            self.assertTrue(ui.success)
            batch_id = ui.code[0].value
            batch = api.get(f"/data/batches/{batch_id}").json()
            self.assertEqual(batch["crop"], "maize")
            self.assertCountEqual([(self.root / row["storage_path"]).read_bytes() for row in batch["images"]], images)
            self.assertEqual([m.value for m in ui.metric], ["1", "2"])
            counts = ui.table[0].value.set_index("Crop")["Images"].to_dict()
            self.assertEqual(counts, {"Cashew": 0, "Cassava": 0, "Maize": 2, "Rice": 0, "Tomato": 0})
            ui.button[1].click().run()
            post.assert_called_once()
            ui.button[0].click().run()
            self.assertFalse(ui.success)
            self.assertIn("already stored", ui.error[0].value)
            self.assertEqual(ui.code[0].value, batch_id)
            self.assertEqual([m.value for m in ui.metric], ["1", "2"])

    def test_dashboard_tracks_latest_job_for_selected_batch_and_shows_failure(self):
        ui = AppTest.from_file("../frontend/app.py", default_timeout=30).run()
        with TestClient(app) as api, patch("httpx.get", side_effect=api.get), patch("httpx.post", side_effect=api.post) as post:
            ui.switch_page("retraining.py").run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.button)
            rice = api.post("/data/upload", headers=OPERATOR_HEADERS, data={"crop": "rice"},
                            files={"files": ("a.jpg", jpeg("green"), "image/jpeg")}).json()["batch_id"]
            maize = api.post("/data/upload", headers=OPERATOR_HEADERS, data={"crop": "maize"},
                             files={"files": ("b.jpg", jpeg("yellow"), "image/jpeg")}).json()["batch_id"]
            ui.run()
            ui.selectbox[0].set_value(rice).run()
            self.assertTrue(ui.button[0].disabled)
            ui.text_input(key="operator_token").set_value(TEST_TOKEN).run()
            ui.button[0].click().run()
            self.assertTrue(ui.button[0].disabled)
            job_id = ui.code[1].value
            self.assertEqual(api.get(f"/train/jobs/{job_id}").json()["batch_id"], rice)
            ui.selectbox[0].set_value(maize).run()
            self.assertEqual([elem.value for elem in ui.code], ["crop_model_v1"])
            with patch("src.worker.train_job", side_effect=ValueError("Image checksum mismatch")), self.assertLogs():
                process_one(app.state.upload_store, self.root)
            ui.selectbox[0].set_value(rice).run()
            self.assertEqual(ui.code[1].value, job_id)
            self.assertEqual(ui.error[0].value, "Image checksum mismatch")
            self.assertFalse(ui.button[0].disabled)
            post.assert_called_once()

    async def test_summary_counts_committed_images_across_restart(self):
        zeros = {"cashew": 0, "cassava": 0, "maize": 0, "rice": 0, "tomato": 0}
        expected = {"total_batches": 2, "total_images": 3,
                    "images_by_crop": {**zeros, "rice": 2, "maize": 1}}
        rice_leaf = jpeg("green")
        async with lifespan(app), self.client() as client:
            empty = await client.get("/data/summary")
            self.assertEqual(empty.status_code, 200)
            self.assertEqual(empty.json(), {"total_batches": 0, "total_images": 0,
                                            "images_by_crop": zeros})
            self.assertEqual((await self.upload(client, rice_leaf, jpeg("blue"))).status_code, 201)
            self.assertEqual((await self.upload(client, jpeg("yellow"), crop="maize")).status_code, 201)
            self.assertEqual((await client.get("/data/summary")).json(), expected)
            self.assertEqual((await self.upload(client, rice_leaf, crop="tomato")).status_code, 409)
            self.assertEqual((await client.get("/data/summary")).json(), expected)

        async with lifespan(app), self.client() as client:
            self.assertEqual((await client.get("/data/summary")).json(), expected)

    async def test_retraining_queue_and_failure_are_visible_across_restart(self):
        async with lifespan(app), self.client() as client:
            missing = str(uuid4())
            self.assertEqual((await client.post("/train/retrain", json={"batch_id": missing})).status_code, 404)
            self.assertEqual((await client.get(f"/train/jobs/{missing}")).status_code, 404)
            batch = (await self.upload(client, jpeg("green"))).json()
            body = {"batch_id": batch["batch_id"]}
            response = await client.post("/train/retrain", json=body)
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["status"], "queued")
            url = response.headers["Location"]
            self.assertEqual((await client.post("/train/retrain", json=body)).headers["Location"], url)

        async with lifespan(app), self.client() as client:
            store = app.state.upload_store
            with psycopg.connect(self.database) as owner:
                owner.execute("SELECT pg_advisory_lock(72545109)")
                self.assertFalse(process_one(store, self.root))
                self.assertEqual((await client.get(url)).json()["status"], "queued")
            with patch("src.worker.train_job", side_effect=ValueError("Image checksum mismatch")), self.assertLogs():
                self.assertTrue(process_one(store, self.root))
            failed = (await client.get(url)).json()
            self.assertEqual((failed["status"], failed["error"]), ("failed", "Image checksum mismatch"))
            self.assertIsNotNone(failed["finished_at"])
            self.assertNotEqual((await client.post("/train/retrain", json=body)).headers["Location"], url)

    async def test_worker_recovers_published_candidate_after_interruption(self):
        async with lifespan(app), self.client() as client:
            batch_id = (await self.upload(client, jpeg("green"))).json()["batch_id"]
            response = await client.post("/train/retrain", json={"batch_id": batch_id})
            job_id = response.json()["job_id"]
            with psycopg.connect(self.database) as connection:
                connection.execute("UPDATE leafvision_training_job SET status='running' WHERE job_id=%s", (job_id,))
            run = self.root / job_id
            run.mkdir()
            (run / "candidate.keras").write_bytes(b"published test artifact")
            report = {"batch_id": batch_id, "checkpoint_sha256": "baseline",
                      "candidate_sha256": hashlib.sha256(b"published test artifact").hexdigest(),
                      "baseline": {"correct": 3}, "candidate_metrics": {"correct": 4},
                      "max_reload_score_difference": 0}
            (run / "report.json").write_text(json.dumps(report))
            with patch("src.worker.retrain") as training:
                self.assertTrue(process_one(app.state.upload_store, self.root))
                training.assert_not_called()
            finished = (await client.get(response.headers["Location"])).json()
            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["result"]["candidate_metrics"], {"correct": 4})
            self.assertFalse(process_one(app.state.upload_store, self.root))

    def test_dashboard_promotion_and_rollback_use_the_displayed_version(self):
        ui = AppTest.from_file("../frontend/app.py", default_timeout=30).run()
        with (
            TestClient(app) as api,
            patch("httpx.get", side_effect=api.get),
            patch("httpx.post", side_effect=api.post) as post,
            patch("src.api.main.ARTIFACTS", self.root),
            patch("src.registry.load_crop_model"),
        ):
            batch_id = api.post("/data/upload", headers=OPERATOR_HEADERS, data={"crop": "rice"},
                                files={"files": ("a.jpg", jpeg("green"), "image/jpeg")}).json()["batch_id"]
            job_id = api.post("/train/retrain", headers=OPERATOR_HEADERS, json={"batch_id": batch_id}).json()["job_id"]
            candidate = self.root / "retraining" / job_id / "candidate.keras"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"candidate")
            metrics = {"images": 10, "support": {"rice": 10}, "accuracy": .9, "macro_precision": .9,
                       "macro_recall": .9, "macro_f1": .9, "mean_cross_entropy": .3}
            result = {"checkpoint_sha256": V1_SOURCE, "candidate_version": job_id,
                      "candidate_sha256": hashlib.sha256(b"candidate").hexdigest(),
                      "baseline": metrics, "candidate_metrics": dict(metrics)}
            with patch("src.worker.train_job", return_value=result):
                process_one(app.state.upload_store, self.root)

            ui.switch_page("retraining.py").run()
            promote = f"promote_{job_id}"
            self.assertTrue(ui.button(key=promote).disabled)
            ui.text_input(key="operator_token").set_value(TEST_TOKEN).run()
            ui.button(key=promote).click().run()
            self.assertEqual(ui.error[0].value, "Validation macro F1 must improve")
            self.assertEqual(ui.code[0].value, "crop_model_v1")
            result["candidate_metrics"]["macro_f1"] = .91
            with psycopg.connect(self.database) as connection:
                connection.execute("UPDATE leafvision_training_job SET result=%s WHERE job_id=%s", (Jsonb(result), job_id))
            ui.run()
            ui.button(key=promote).click().run()
            self.assertEqual(ui.code[0].value, job_id)
            self.assertTrue(ui.button(key=promote).disabled)

            ui.text_input(key="operator_token").set_value("").run()
            self.assertTrue(ui.button(key="rollback").disabled)
            ui.text_input(key="operator_token").set_value(TEST_TOKEN).run()

            # A different client restores the baseline while this browser still displays the candidate.
            api.post("/model/rollback", headers=OPERATOR_HEADERS, json={"expected_active": job_id}).raise_for_status()
            ui.button(key="rollback").click().run()
            self.assertEqual(post.call_args.kwargs["json"], {"expected_active": job_id})
            self.assertIn("Active version is crop_model_v1", ui.error[0].value)
            self.assertEqual(ui.code[0].value, "crop_model_v1")
            ui.button(key=promote).click().run()
            ui.button(key="rollback").click().run()
            self.assertEqual(ui.code[0].value, "crop_model_v1")
            self.assertFalse(ui.error)
            self.assertFalse(ui.exception)

    async def test_only_verified_candidate_switches_and_next_worker_uses_it(self):
        baseline = {"expected_active": "crop_model_v1"}
        async with lifespan(app), self.client() as client:
            batch_id = (await self.upload(client, jpeg("green"))).json()["batch_id"]
            job = await client.post("/train/retrain", json={"batch_id": batch_id})
            version = job.json()["job_id"]
            promote = f"/model/versions/{version}/promote"
            self.assertEqual((await client.post(promote, json=baseline)).status_code, 409)
            path = self.root / "retraining" / version / "candidate.keras"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"candidate")
            before = {"images": 10, "support": {"rice": 10}, "accuracy": .9,
                      "macro_f1": .9, "mean_cross_entropy": .3}
            result = {"checkpoint_sha256": V1_SOURCE, "candidate_sha256": hashlib.sha256(b"candidate").hexdigest(),
                      "baseline": before, "candidate_metrics": dict(before)}
            def finish():
                with psycopg.connect(self.database) as conn:
                    conn.execute("UPDATE leafvision_training_job SET status='succeeded', result=%s WHERE job_id=%s",
                                 (Jsonb(result), version))
            with patch("src.api.main.ARTIFACTS", self.root), patch("src.registry.load_crop_model"):
                finish()
                self.assertEqual((await client.post(promote, json=baseline)).status_code, 409)
                result["candidate_metrics"]["macro_f1"] = .91
                finish()
                path.write_bytes(b"damaged")
                self.assertEqual((await client.post(promote, json=baseline)).status_code, 409)
                self.assertEqual((await client.get("/model/info")).json()["model_version"], "crop_model_v1")
                path.write_bytes(b"candidate")
                self.assertEqual((await client.post(promote, json=baseline)).json()["model_version"], version)
                self.assertEqual(training_checkpoint(self.database, self.root / "retraining"), path)
                self.assertEqual((await client.post("/model/rollback", json=baseline)).status_code, 409)
                self.assertEqual((await client.get("/model/info")).json()["model_version"], version)
                rollback = await client.post("/model/rollback", json={"expected_active": version})
                self.assertEqual(rollback.json()["model_version"], "crop_model_v1")

    async def test_batch_survives_restart_and_rejected_batches_leave_no_trace(self):
        first, second, third = jpeg("green"), jpeg("blue"), jpeg("yellow")
        async with lifespan(app), self.client() as client:
            response = await self.upload(client, first, second)
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(response.json()["image_count"], 2)
            batch_id = response.json()["batch_id"]

        async with lifespan(app), self.client() as client:
            response = await client.get(f"/data/batches/{batch_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["crop"], "rice")
            by_hash = {hashlib.sha256(data).hexdigest(): data for data in (first, second)}
            for image in response.json()["images"]:
                data = by_hash[image["sha256"]]
                self.assertEqual((self.root / image["storage_path"]).read_bytes(), data)
                self.assertEqual(image["size_bytes"], len(data))

            self.assertEqual((await self.upload(client, third, b"bad jpeg")).status_code, 422)
            self.assertEqual((await self.upload(client, third, crop="wheat")).status_code, 422)
            self.assertEqual((await self.upload(client, third, third)).status_code, 409)
            self.assertEqual((await self.upload(client, third, first, crop="maize")).status_code, 409)
            self.assertEqual((await client.get(f"/data/batches/{uuid4()}")).status_code, 404)

        with psycopg.connect(self.database) as connection:
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM leafvision_upload_batch"
            ).fetchone()[0], 1)
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM leafvision_uploaded_image"
            ).fetchone()[0], 2)
        self.assertEqual([path.name for path in self.root.iterdir()], [batch_id])

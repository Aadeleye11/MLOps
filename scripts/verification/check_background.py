import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import embedded_postgres
import httpx

headers = {"Authorization": f"Bearer {os.environ['LEAFVISION_OPERATOR_TOKEN']}"}
artifacts = Path("artifacts").resolve()
batch_id = "0f02ff9d-2e51-42a4-a24f-b52879cff427"
with tempfile.TemporaryDirectory() as tmp, embedded_postgres.get_server(artifacts / "retraining-inputs/pgdata") as db:
    env = {**os.environ, "LEAFVISION_DATABASE_URL": db.get_uri()}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with open(Path(tmp) / "api.log", "w+") as log, httpx.Client(base_url=base, trust_env=False, timeout=20) as api:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(port)],
                                  env=env, stdout=log, stderr=log)
        worker = None
        try:
            for i in range(150):
                if server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    if api.get("/health").is_success:
                        break
                except httpx.ConnectError:
                    pass
                time.sleep(0.2)
            start = time.monotonic()
            request = api.post("/train/retrain", headers=headers, json={"batch_id": batch_id})
            request.raise_for_status()
            assert request.status_code == 202 and request.json()["status"] == "queued"
            url = request.headers["Location"]
            print("Accepted", request.json(), "in", round(time.monotonic() - start, 3), "s", flush=True)
            assert api.post("/train/retrain", headers=headers, json={"batch_id": batch_id}).headers["Location"] == url
            worker = subprocess.Popen([sys.executable, "-m", "src.worker", "--once"], env=env)
            stages = set()
            for i in range(1200):
                reply = api.get(url)
                reply.raise_for_status()
                job = reply.json()
                stage = job["stage"] or job["status"]
                if stage not in stages:
                    print(stage, flush=True)
                    stages.add(stage)
                    if stage == "validating":
                        with ZipFile(artifacts / "archives/rice_original.zip") as archive:
                            data = archive.read("Rice Dataset/Original Dataset/Healthy/HL_423.jpg")
                        result = api.post("/predict", files={"file": ("rice.jpg", data, "image/jpeg")})
                        result.raise_for_status()
                        print("Concurrent prediction", result.json(), flush=True)
                        assert result.json()["crop"] == "rice"
                if job["status"] in ("succeeded", "failed"):
                    break
                if worker.poll() not in (None, 0):
                    raise RuntimeError("Worker exited early")
                time.sleep(1)
            assert job["status"] == "succeeded", job
            assert "validating" in stages
            report = json.loads((artifacts / "retraining" / job["job_id"] / "report.json").read_text())
            assert report["candidate_metrics"] == job["result"]["candidate_metrics"]
            assert report["baseline"] == job["result"]["baseline"]
            assert report["candidate_sha256"] == job["result"]["candidate_sha256"]
            print(json.dumps(job), flush=True)
            assert worker.wait(timeout=30) == 0
        finally:
            for process in (worker, server):
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)

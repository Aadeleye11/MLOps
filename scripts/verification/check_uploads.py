import hashlib
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import embedded_postgres
import httpx
import psycopg


@contextmanager
def running_api(environment, logs):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-m", "uvicorn", "src.api.main:app",
               "--host", "127.0.0.1", "--port", str(port)]
    with logs.open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(logs.read_text())
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError(logs.read_text())
                yield client
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)


with tempfile.TemporaryDirectory(prefix="leafvision-db-") as directory:
    root = Path(directory)
    with embedded_postgres.get_server(root / "pgdata") as postgres:
        database = postgres.get_uri()
        env = {**os.environ, "LEAFVISION_TEST_DATABASE_URL": database}
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                       env=env, check=True)
        env.update(LEAFVISION_DATABASE_URL=database, LEAFVISION_UPLOAD_ROOT=str(root / "uploads"),
                   LEAFVISION_OPERATOR_TOKEN=secrets.token_hex(32))
        headers = {"Authorization": f"Bearer {env['LEAFVISION_OPERATOR_TOKEN']}"}
        fixtures = Path("artifacts/retraining-inputs")
        entries = [("files", (name, (fixtures / name).read_bytes(), "image/jpeg"))
                   for name in ("tomato-259.jpg", "tomato-308.jpg")]
        with running_api(env, root / "before.log") as client:
            sample = Path("artifacts/HL_423.jpg")
            prediction = client.post("/predict", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")})
            assert prediction.raise_for_status().json()["crop"] == "rice"
            denied = client.post("/data/upload", files=entries, data={"crop": "tomato"})
            assert denied.status_code == 401
            response = client.post("/data/upload", headers=headers, files=entries, data={"crop": "tomato"})
            assert response.status_code == 201, response.text
            batch_id = response.json()["batch_id"]
            job = client.post("/train/retrain", headers=headers, json={"batch_id": batch_id})
            assert job.status_code == 202 and job.json()["status"] == "queued"
            print("Public rice prediction passed. Operator saved two JPEGs and queued a retraining job.")
        with running_api(env, root / "after.log") as client:
            response = client.get(f"/data/batches/{batch_id}")
            assert response.status_code == 200, response.text
            batch = response.json()
            assert batch["crop"] == "tomato" and len(batch["images"]) == 2
            by_hash = {hashlib.sha256(image).hexdigest(): image for _, (_, image, _) in entries}
            for image in batch["images"]:
                assert (root / "uploads" / image["storage_path"]).read_bytes() == by_hash[image["sha256"]]
        with psycopg.connect(database) as connection:
            print("PostgreSQL", connection.execute("SHOW server_version").fetchone()[0])
        print("Live upload passed. Both JPEGs and both metadata records survived a Uvicorn process restart.")

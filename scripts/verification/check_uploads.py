import hashlib
import io
import os
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
from PIL import Image


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
                for _ in range(250):
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
        env.update(LEAFVISION_DATABASE_URL=database, LEAFVISION_UPLOAD_ROOT=str(root / "uploads"))
        images = []
        for color in ("green", "blue"):
            buffer = io.BytesIO()
            Image.new("RGB", (32, 24), color).save(buffer, format="JPEG")
            images.append(buffer.getvalue())
        entries = [("files", (f"leaf-{i}.jpg", data, "image/jpeg")) for i, data in enumerate(images)]
        with running_api(env, root / "before.log") as client:
            response = client.post("/data/upload", files=entries, data={"crop": "rice"})
            assert response.status_code == 201, response.text
            batch_id = response.json()["batch_id"]
        with running_api(env, root / "after.log") as client:
            response = client.get(f"/data/batches/{batch_id}")
            assert response.status_code == 200, response.text
            batch = response.json()
            assert batch["crop"] == "rice" and len(batch["images"]) == 2
            by_hash = {hashlib.sha256(image).hexdigest(): image for image in images}
            for image in batch["images"]:
                assert (root / "uploads" / image["storage_path"]).read_bytes() == by_hash[image["sha256"]]
        with psycopg.connect(database) as connection:
            print("PostgreSQL", connection.execute("SHOW server_version").fetchone()[0])
        print("Live upload passed. Both JPEGs and both metadata records survived a Uvicorn process restart.")

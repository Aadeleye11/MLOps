import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import embedded_postgres
import httpx

headers = {"Authorization": f"Bearer {os.environ['LEAFVISION_OPERATOR_TOKEN']}"}
root = Path("artifacts").resolve()
candidate = "3bd92605-8a72-44dd-9810-3b2d235234c1"
baseline = "crop_model_v1"
image = (root / "retraining-inputs/tomato-308.jpg").read_bytes()


@contextmanager
def replica(database):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile("w+") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(port)],
                                   env={**os.environ, "LEAFVISION_DATABASE_URL": database}, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                for _ in range(200):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(.2)
                yield client
        finally:
            process.terminate()
            process.wait(timeout=20)


def verify(clients, version, score):
    for client in clients:
        assert client.get("/model/info").json()["model_version"] == version
        response = client.post("/predict", files={"file": ("leaf.jpg", image, "image/jpeg")})
        response.raise_for_status()
        result = response.json()
        assert result["model_version"] == version and result["crop"] == "tomato", result
        assert abs(result["score"] - score) < 1e-6, result
        print("Replica", client.base_url.port, result, flush=True)


with embedded_postgres.get_server(root / "retraining-inputs/pgdata") as db, ExitStack() as stack:
    clients = [stack.enter_context(replica(db.get_uri())) for _ in range(2)]
    verify(clients, baseline, 0.92766124)
    control = clients[0]
    response = control.post(f"/model/versions/{candidate}/promote", headers=headers, json={"expected_active": baseline})
    response.raise_for_status()
    try:
        verify(clients, candidate, 0.95196134)
        assert control.post("/model/rollback", headers=headers, json={"expected_active": baseline}).status_code == 409
    finally:
        response = control.post("/model/rollback", headers=headers, json={"expected_active": candidate})
        response.raise_for_status()
    verify(clients, baseline, 0.92766124)

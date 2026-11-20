import os
import subprocess
import sys
from pathlib import Path

from embedded_postgres import get_server
from fastapi.testclient import TestClient

root = Path.cwd()
sys.path.insert(0, str(root))
from src.api.main import app

with get_server(root / "artifacts" / "retraining-inputs" / "pgdata") as db:
    os.environ["LEAFVISION_DATABASE_URL"] = db.get_uri()
    with TestClient(app) as client:
        files = [("files", (p.name, p.read_bytes(), "image/jpeg"))
                 for p in sorted(Path("artifacts/retraining-inputs").glob("tomato-*.jpg"))]
        batch = client.post("/data/upload", files=files, data={"crop": "tomato"})
        batch.raise_for_status()
        print("Upload accepted", batch.json(), flush=True)
        batch_id = batch.json()["batch_id"]
    subprocess.run([sys.executable, "-m", "src.retraining", "--batch-id", batch_id], check=True)

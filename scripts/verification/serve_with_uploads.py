import os
from pathlib import Path

from embedded_postgres import get_server
import uvicorn

root = Path.cwd()
with get_server(root / "artifacts/retraining-inputs/pgdata") as db:
    os.environ["LEAFVISION_DATABASE_URL"] = db.get_uri()
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, app_dir=str(root))

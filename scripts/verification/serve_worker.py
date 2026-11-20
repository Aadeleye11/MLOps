import os
from pathlib import Path
from embedded_postgres import get_server
from src.worker import main

with get_server(Path("artifacts/retraining-inputs/pgdata").resolve()) as db:
    os.environ["LEAFVISION_DATABASE_URL"] = db.get_uri()
    main()

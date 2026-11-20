import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx

root = Path.cwd()
with tempfile.TemporaryDirectory(prefix="leafvision-api-package-") as temp:
    app = Path(temp)
    files = [*root.glob("src/*.py"), *root.glob("src/api/*.py"),
             root/"scripts/audit_image_hashes.py", root/"scripts/prepare_splits.py",
             root/"models/crop_model_v1.h5"]
    for source in files:
        target = app / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    env = dict(os.environ)
    for name in ("PYTHONPATH", "LEAFVISION_DATABASE_URL", "LEAFVISION_MODEL_PATH", "LEAFVISION_UPLOAD_ROOT"):
        env.pop(name, None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (app/"server.log").open("w+") as log:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app",
                                 "--host", "127.0.0.1", "--port", str(port)],
                                cwd=app, env=env, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
                for _ in range(120):
                    if proc.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(.25)
                info = client.get("/model/info").json()
                response = client.post("/predict", files={"file": ("HL_423.jpg", (root/"artifacts/HL_423.jpg").read_bytes(), "image/jpeg")})
                response.raise_for_status()
                prediction = response.json()
                assert prediction["model_version"] == info["model_version"] == "crop_model_v1"
                assert prediction["crop"] == "rice" and abs(prediction["score"]-0.99999988) < 1e-6
                print(json.dumps(prediction))
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)

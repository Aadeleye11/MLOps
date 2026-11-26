import os
import math
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path.cwd() / "scripts"))
from check_replica_routing import check

headers = {"Authorization": f"Bearer {os.environ['LEAFVISION_OPERATOR_TOKEN']}"}
baseline = "crop_model_v1"
candidate = "7ea0b21d-b4e2-443c-91bf-6b513c1d35b5"
image = Path("artifacts/retraining-inputs/tomato-308.jpg")

with httpx.Client(base_url="http://127.0.0.1:18000", timeout=90, trust_env=False) as client:
    def active():
        return client.get("/model/info").raise_for_status().json()["model_version"]

    assert active() == baseline
    job = client.get(f"/train/jobs/{candidate}").raise_for_status().json()
    assert job["status"] == "succeeded"
    before = check(client, 3, image, "tomato", baseline)
    try:
        response = client.post(f"/model/versions/{candidate}/promote", headers=headers, json={"expected_active": baseline})
        assert response.raise_for_status().json()["sha256"] == job["result"]["candidate_sha256"]
        promoted = check(client, 3, image, "tomato", candidate)
        assert not math.isclose(before, promoted, abs_tol=1e-7)
    finally:
        if active() == candidate:
            restored = client.post("/model/rollback", headers=headers, json={"expected_active": candidate})
            assert restored.raise_for_status().json()["model_version"] == baseline
    after = check(client, 3, image, "tomato", baseline)
    assert math.isclose(before, after, abs_tol=1e-7)

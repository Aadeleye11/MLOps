"""Exercise LeafVision's labeled rice prediction through its local Compose proxy."""

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

from locust import HttpUser, constant, events, task
from locust.stats import PERCENTILES_TO_REPORT, StatsCSV


PHOTO = Path(__file__).resolve().parents[1] / "artifacts" / "HL_423.jpg"
JPEG = PHOTO.read_bytes()
if hashlib.sha256(JPEG).hexdigest() != "2878ab34de6ae687a4bd273acbc28e5fe0fd8f95ada1d8185a9102b0dd1bda3b":
    raise ValueError(f"Expected the original labeled rice photo at {PHOTO}")

backends = Counter()


class LeafUser(HttpUser):
    host = "http://127.0.0.1:18090"
    wait_time = constant(0)

    @task
    def identify_rice(self):
        with self.client.post(
            "/predict", files={"file": (PHOTO.name, JPEG, "image/jpeg")},
            timeout=30, catch_response=True, allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Prediction HTTP status {response.status_code}")
                return
            try:
                result = response.json()
                valid = (result["crop"] == "rice" and result["model_version"] == "crop_model_v1"
                         and math.isclose(result["score"], 0.99999988, abs_tol=1e-5))
            except (KeyError, ValueError, TypeError):
                valid = False
            backend = response.headers.get("X-LeafVision-Replica")
            if valid and backend:
                backends[backend] += 1
            else:
                response.failure("Unexpected crop, model, score, or missing proxy backend")


@events.test_start.add_listener
def reset_counts(environment, **kwargs):
    backends.clear()


@events.test_stop.add_listener
def save_completed_requests(environment, **kwargs):
    """Snapshot after in-flight requests finish, including the last partial CSV interval."""
    prefix = environment.parsed_options.csv_prefix
    if prefix:
        with open(f"{prefix}_completed.csv", "w", newline="") as stream:
            StatsCSV(environment, PERCENTILES_TO_REPORT).requests_csv(csv.writer(stream))
        with open(f"{prefix}_backends.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["backend", "validated_responses"])
            writer.writerows(sorted(backends.items()))

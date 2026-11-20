"""Verify prediction routing and model consistency through the local Compose proxy."""

import argparse
import math
from collections import Counter
from pathlib import Path

import httpx


def check(client, replicas, image, crop, version):
    photo = (image.name, image.read_bytes(), "image/jpeg")
    counts = Counter()
    first_score = None
    for _ in range(3 * replicas):
        response = client.post("/predict", files={"file": photo})
        response.raise_for_status()
        backend = response.headers["X-LeafVision-Replica"]
        result = response.json()
        if (result["crop"], result["model_version"]) != (crop, version):
            raise ValueError(f"{backend} returned an unexpected prediction: {result}")
        score = float(result["score"])
        if first_score is None:
            first_score = score
        if not math.isfinite(score) or not math.isclose(score, first_score, abs_tol=1e-7):
            raise ValueError(f"Inconsistent score on {backend}: {score} versus {first_score}")
        counts[backend] += 1
    if len(counts) != replicas:
        raise ValueError(f"Expected {replicas} distinct backends, saw {dict(counts)}")
    print(f"{version} | {crop} | score {first_score:.9f} | requests {dict(sorted(counts.items()))}")
    return first_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--image", type=Path, default=Path("artifacts/HL_423.jpg"))
    parser.add_argument("--crop", default="rice")
    parser.add_argument("--version", default="crop_model_v1")
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, trust_env=False, timeout=60) as client:
        check(client, args.replicas, args.image, args.crop, args.version)

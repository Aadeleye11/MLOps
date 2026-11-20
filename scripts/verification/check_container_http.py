from pathlib import Path
import time

import httpx

jpeg = Path("artifacts/HL_423.jpg").read_bytes()
with httpx.Client(timeout=20, trust_env=False) as client:
    container = "http://127.0.0.1:18000"
    for _ in range(60):
        try:
            health = client.get(f"{container}/health")
            if health.is_success and health.json()["status"] == "ready":
                break
        except httpx.TransportError:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("Container did not become ready on port 18000")
    info = client.get(f"{container}/model/info")
    info.raise_for_status()
    assert info.json()["input_shape"] == [128, 128, 3] and len(info.json()["classes"]) == 5
    scores = []
    for endpoint in ("http://127.0.0.1:8000", container):
        response = client.post(f"{endpoint}/predict", files={"file": ("HL_423.jpg", jpeg, "image/jpeg")})
        response.raise_for_status()
        output = response.json()
        assert output["crop"] == "rice" and output["model_version"] == "crop_model_v1", output
        scores.append(output["score"])
        print(endpoint, output, flush=True)
    assert abs(scores[0] - scores[1]) <= 1e-6
    assert client.post(f"{container}/predict", files={"file": ("bad.jpg", b"invalid", "image/jpeg")}).status_code == 422
    print("Health, schema, baseline parity, and invalid JPEG checks passed.")

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

root = Path.cwd()
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30, trust_env=False) as client:
    client.get("http://127.0.0.1:8501/monitoring").raise_for_status()
    def summary():
        response = client.get("/monitoring/summary")
        response.raise_for_status()
        return response.json()
    start = summary()
    app = AppTest.from_file(str(root / "frontend/app.py"), default_timeout=30).run()
    app.switch_page("monitoring.py").run()
    assert not app.exception and not app.error
    if start["requests"] == 0:
        assert len(app.metric) == 3 and start["mean_ms"] is None
    data = (root / "artifacts/HL_423.jpg").read_bytes()
    response = client.post("/predict", files={"file": ("rice.jpg", data, "image/jpeg")})
    response.raise_for_status()
    assert response.json()["crop"] == "rice"
    assert client.post("/predict", files={"file": ("invalid.jpg", b"invalid", "image/jpeg")}).status_code == 422
    assert client.post("/predict").status_code == 422
    app.run()
    end = summary()
    assert end["requests"] - start["requests"] == 3 and end["failures"] - start["failures"] == 2
    assert end["uptime_seconds"] >= start["uptime_seconds"]
    assert end["model_version"] == response.json()["model_version"] == app.code[0].value
    assert not app.exception and not app.error
    shown = {m.label: m.value for m in app.metric}
    assert shown["Prediction requests"] == str(end["requests"]) and shown["Failed requests"] == str(end["failures"])
    assert shown["Mean"] == f'{end["mean_ms"]:,.1f} ms' and shown["95th percentile"] == f'{end["p95_ms"]:,.1f} ms'
    assert app.table[0].value["HTTP status"].tolist()[:2] == ["422", "422"]
    print("Live monitor", shown, "failure HTTP statuses", [f["status"] for f in end["recent_failures"]])

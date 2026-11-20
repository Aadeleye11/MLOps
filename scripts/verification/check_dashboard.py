from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

root = Path.cwd()
data = (root / "artifacts/HL_423.jpg").read_bytes()
photo = ("HL_423.jpg", data, "image/jpeg")
with httpx.Client(timeout=30, trust_env=False) as client:
    ui = client.get("http://127.0.0.1:8501/")
    ui.raise_for_status()
    health = client.get("http://127.0.0.1:8501/_stcore/health")
    health.raise_for_status()
    response = client.post("http://127.0.0.1:8000/predict", files={"file": photo})
    response.raise_for_status()
    prediction = response.json()
app = AppTest.from_file(str(root / "frontend/app.py"), default_timeout=30).run()
assert not app.exception, app.exception
app.file_uploader[0].set_value(photo).run()
app.button[0].click().run()
assert not app.exception, app.exception
assert not app.error, app.error
assert prediction["crop"] == "rice" and app.metric[0].value.lower() == prediction["crop"]
assert app.metric[1].value == format(prediction["score"], ".8f")
assert app.text[0].value == prediction["model_version"]
print("Dashboard HTTP", ui.status_code, "health", health.text)
print([(m.label, m.value) for m in app.metric], app.text[0].value)
app.file_uploader[0].set_value(("broken.jpg", b"broken JPEG", "image/jpeg")).run()
assert not app.metric
app.button[0].click().run()
assert not app.exception and app.error and not app.metric
print("Invalid JPEG:", app.error[0].value)

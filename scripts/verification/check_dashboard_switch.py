import os
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

headers = {"Authorization": f"Bearer {os.environ['LEAFVISION_OPERATOR_TOKEN']}"}
root = Path.cwd()
photo = (root / "artifacts/retraining-inputs/tomato-308.jpg").read_bytes()
batch = "0f02ff9d-2e51-42a4-a24f-b52879cff427"
with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=30) as client:
    ui = AppTest.from_file(str(root / "frontend/app.py"), default_timeout=45).run()
    ui.switch_page("retraining.py").run()
    ui.text_input(key="operator_token").set_value(os.environ["LEAFVISION_OPERATOR_TOKEN"]).run()
    ui.selectbox[0].set_value(batch).run()
    assert not ui.exception and not ui.error, (ui.exception, ui.error)
    info = client.get("/model/info").json()
    assert info["model_version"] == info["baseline_version"] == "crop_model_v1"
    candidate = ui.code[2].value
    def check(version):
        assert ui.code[0].value == version and not ui.exception and not ui.error, (ui.exception, ui.error)
        r = client.post("/predict", files={"file": ("tomato.jpg", photo, "image/jpeg")})
        r.raise_for_status()
        result = r.json()
        assert result["crop"] == "tomato" and result["model_version"] == version, result
        print(version, result["crop"], result["score"], flush=True)
        return result["score"]
    initial = check("crop_model_v1")
    try:
        ui.button(key=f"promote_{candidate}").click().run()
        assert check(candidate) != initial
        assert ui.button(key=f"promote_{candidate}").disabled
        ui.button(key="rollback").click().run()
        assert check("crop_model_v1") == initial
    finally:
        if client.get("/model/info").json()["model_version"] == candidate:
            client.post("/model/rollback", headers=headers, json={"expected_active": candidate}).raise_for_status()

import os
from pathlib import Path
import json
import time

import httpx
from streamlit.testing.v1 import AppTest

root = Path.cwd()
batch = "0f02ff9d-2e51-42a4-a24f-b52879cff427"
with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=15) as client:
    client.get("http://127.0.0.1:8501/retraining").raise_for_status()
    previous = next(b["latest_job_id"] for b in client.get("/data/batches").json() if b["batch_id"] == batch)
    app = AppTest.from_file(str(root / "frontend/app.py"), default_timeout=30).run()
    app.switch_page("retraining.py").run()
    app.text_input(key="operator_token").set_value(os.environ["LEAFVISION_OPERATOR_TOKEN"]).run()
    assert not app.exception and not app.error, (app.exception, app.error)
    app.selectbox[0].set_value(batch).run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception and not app.error, (app.exception, app.error)
    job_id = app.code[1].value
    assert job_id != previous
    print("Started job", job_id, flush=True)
    seen = set()
    for i in range(600):
        app.run()
        assert not app.exception and not app.error, (app.exception, app.error)
        if app.success:
            break
        for msg in app.info:
            if msg.value not in seen:
                seen.add(msg.value)
                print(msg.value, flush=True)
        assert app.button[0].disabled
        time.sleep(2)
    assert app.success
    assert app.code[1].value == job_id and app.code[2].value == job_id
    job = client.get(f"/train/jobs/{job_id}").json()
    assert job["status"] == "succeeded"
    report = json.loads((root / "artifacts/retraining" / job_id / "report.json").read_text())
    table = app.table[0].value.set_index("Metric")
    for label, key in (("Accuracy","accuracy"), ("Macro precision","macro_precision"),
                       ("Macro recall","macro_recall"), ("Macro F1","macro_f1"), ("Cross-entropy loss","mean_cross_entropy")):
        for column, source in (("Baseline","baseline"), ("Candidate","candidate_metrics")):
            assert table.loc[label,column] == format(report[source][key], ".5f")
    assert app.code[0].value == client.get("/model/info").json()["model_version"] == "crop_model_v1"
    print(job_id, app.code[0].value, table.to_string(), flush=True)

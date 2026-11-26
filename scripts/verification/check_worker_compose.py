import os
import time
from pathlib import Path
from uuid import UUID

import httpx
from streamlit.testing.v1 import AppTest


def clean(app):
    assert not app.exception and not app.error, (app.exception, app.error)


app = AppTest.from_file("/app/frontend/app.py", default_timeout=60).run()
app.switch_page("bulk_upload.py").run()
app.text_input(key="operator_token").set_value(os.environ["LEAFVISION_OPERATOR_TOKEN"]).run()
clean(app)
photos = [Path("/tmp") / name for name in ("tomato-259.jpg", "tomato-308.jpg")]
app.selectbox[0].select("tomato")
app.file_uploader[0].set_value([(p.name, p.read_bytes(), "image/jpeg") for p in photos]).run()
app.button[0].click().run()
clean(app)
assert app.success
batch_id = str(UUID(app.code[0].value))
app.switch_page("retraining.py").run()
clean(app)
assert app.selectbox[0].value == batch_id and app.button[0].label == "Retrain"
app.button[0].click().run()
clean(app)
job_id = str(UUID(app.code[1].value))
print("batch", batch_id, "job", job_id, flush=True)
with httpx.Client(base_url="http://api:8000", timeout=30) as client:
    previous = None
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        job = client.get(f"/train/jobs/{job_id}").raise_for_status().json()
        stage = (job["status"], job["stage"])
        if stage != previous:
            print(stage, flush=True)
            previous = stage
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(2)
    assert job["status"] == "succeeded", (job["status"], job.get("error"))
    app.run()
    clean(app)
    assert app.success and [x.value for x in app.code] == ["crop_model_v1", job_id, job_id]
    table = app.table[0].value.set_index("Metric")
    for label, key in (("Accuracy", "accuracy"), ("Macro precision", "macro_precision"),
                       ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"),
                       ("Cross-entropy loss", "mean_cross_entropy")):
        for column, source in (("Baseline", "baseline"), ("Candidate", "candidate_metrics")):
            assert table.at[label, column] == format(job["result"][source][key], ".5f")
    print(table.to_string(), flush=True)
    print("candidate_sha256", job["result"]["candidate_sha256"], flush=True)

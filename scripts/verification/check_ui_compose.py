from pathlib import Path
from uuid import UUID

import httpx
from streamlit.testing.v1 import AppTest


def fixture(filename):
    return filename, (Path("/tmp") / filename).read_bytes(), "image/jpeg"


app = AppTest.from_file("/app/frontend/app.py", default_timeout=60).run()
assert not app.exception, app.exception
app.file_uploader[0].set_value(fixture("HL_423.jpg")).run()
app.button[0].click().run()
assert not app.error and not app.exception, (app.error, app.exception)
assert app.metric[0].value == "Rice" and app.text[0].value == "crop_model_v1"
print("Prediction:", app.metric[0].value, app.metric[1].value, app.text[0].value)

app.switch_page("dataset_insights.py").run()
assert not app.exception and not app.error, (app.exception, app.error)
app.switch_page("bulk_upload.py").run()
assert [m.value for m in app.metric] == ["0", "0"]
app.selectbox[0].select("tomato")
app.file_uploader[0].set_value([fixture("tomato-259.jpg"), fixture("tomato-308.jpg")]).run()
app.button[0].click().run()
assert not app.exception and not app.error and app.success, (app.exception, app.error)
batch = str(UUID(app.code[0].value))
assert [m.value for m in app.metric] == ["1", "2"]
assert app.table[0].value.set_index("Crop").at["Tomato", "Images"] == 2
receipt = httpx.get(f"http://api:8000/data/batches/{batch}").raise_for_status().json()
assert receipt["crop"] == "tomato" and len(receipt["images"]) == 2
print("Saved batch:", batch, "1 batch / 2 tomato images")

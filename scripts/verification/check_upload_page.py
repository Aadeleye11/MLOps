from pathlib import Path
import httpx
from streamlit.testing.v1 import AppTest

with httpx.Client(trust_env=False, timeout=10) as client:
    client.get("http://127.0.0.1:8501/labeled-uploads").raise_for_status()
    response = client.get("http://127.0.0.1:8000/data/summary")
    response.raise_for_status()
    expected = response.json()
app = AppTest.from_file(str(Path("frontend/app.py").resolve()), default_timeout=15).run()
app.switch_page("bulk_upload.py").run()
assert not app.exception and not app.warning and app.button[0].disabled
assert [m.value for m in app.metric] == [str(expected["total_batches"]), str(expected["total_images"])]
counts = app.table[0].value.set_index("Crop")["Images"].to_dict()
assert counts == {key.capitalize(): count for key, count in expected["images_by_crop"].items()}
print("Verified live saved counts", counts)

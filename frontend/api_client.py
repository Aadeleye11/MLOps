"""Connection settings and error messages shared by dashboard pages."""

import os


def api_url(path):
    base = os.environ.get("LEAFVISION_API_URL", "http://127.0.0.1:8000").rstrip("/")
    return f"{base}{path}"


def error_detail(response):
    if response.status_code in (404, 409, 413, 422, 503):
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            return body["detail"]
    return f"The API returned HTTP {response.status_code}."

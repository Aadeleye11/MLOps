"""Live operational measurements for a LeafVision API process."""

from datetime import timedelta, timezone
from http import HTTPStatus

import httpx
import streamlit as st

from frontend.api_client import api_url, error_detail
from src.schemas import MonitoringSummary


@st.fragment(run_every=5)
def show_metrics():
    try:
        response = httpx.get(api_url("/monitoring/summary"), timeout=5)
        response.raise_for_status()
        stats = MonitoringSummary.model_validate_json(response.content)
    except httpx.HTTPStatusError as error:
        st.error(error_detail(error.response))
        return
    except (httpx.RequestError, ValueError):
        st.error("Monitoring data is unavailable. Check the API connection.")
        return

    if stats.model_version is None:
        st.warning("The API is running, but the serving model is unavailable.")
    else:
        st.caption("Serving model")
        st.code(stats.model_version, language=None)

    uptime, requests, failures = st.columns(3)
    uptime.metric("API uptime", str(timedelta(seconds=int(stats.uptime_seconds))))
    requests.metric("Prediction requests", f"{stats.requests:,}")
    failures.metric("Failed requests", f"{stats.failures:,}")
    if stats.requests:
        st.caption(f"{stats.failures / stats.requests:.1%} of prediction requests returned an HTTP error.")

    st.subheader("Prediction latency")
    if stats.latency_samples:
        avg, p95 = st.columns(2)
        avg.metric("Mean", f"{stats.mean_ms:,.1f} ms")
        p95.metric("95th percentile", f"{stats.p95_ms:,.1f} ms")
        st.caption(f"Server-side duration across the latest {stats.latency_samples} successful "
                   f"requests (window of {stats.latency_window}).")
    else:
        st.write("Latency appears after the first successful prediction.")

    st.subheader("Recent failures")
    if stats.recent_failures:
        st.table([{"Time (UTC)": failure.time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                   "HTTP status": str(failure.status),
                   "Meaning": HTTPStatus(failure.status).phrase if failure.status in HTTPStatus else "HTTP error"}
                  for failure in stats.recent_failures], hide_index=True)
    else:
        st.write("No prediction failures recorded since startup.")


st.title("Monitoring")
st.caption("Measurements for this API process since startup. Updates every five seconds.")
show_metrics()

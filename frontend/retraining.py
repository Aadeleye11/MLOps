"""Start and inspect a retraining job for a saved upload batch."""

import httpx
import streamlit as st

from frontend.api_client import api_url, error_detail, operator_headers


STAGES = {"preparing_data": "Preparing data", "training": "Training", "validating": "Validating candidate"}
METRICS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
    "mean_cross_entropy": "Cross-entropy loss",
}


def read_api(path):
    try:
        response = httpx.get(api_url(path), timeout=10)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        st.error(error_detail(error.response))
    except (httpx.RequestError, ValueError):
        st.error("Could not read from the API. Refresh to try again.")
    return None


def switch_model(path, expected_active, success_message):
    # Callback arguments belong to the screen the operator actually clicked.
    try:
        response = httpx.post(api_url(path), headers=operator_headers(),
                              json={"expected_active": expected_active}, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        notice = (False, error_detail(error.response))
    except httpx.RequestError:
        notice = (False, "Could not confirm the switch. Check the serving version before retrying.")
    else:
        notice = (True, success_message)
    st.session_state["model_switch_notice"] = notice


def show_result(job, model_info):
    if job["status"] == "failed":
        st.error(job["error"] or "Retraining failed. Check the worker log.")
        return
    result = job["result"]
    before, after = result["baseline"], result["candidate_metrics"]
    st.success("Training and validation completed.")
    st.caption("Candidate version")
    st.code(result["candidate_version"], language=None)
    st.subheader("Validation results")
    st.write(f"Both models were evaluated on {after['images']:,} validation images.")
    st.table([{"Metric": label, "Baseline": f"{before[key]:.5f}", "Candidate": f"{after[key]:.5f}"}
              for key, label in METRICS.items()], hide_index=True)
    if model_info:
        active = model_info["model_version"]
        st.caption("Promotion requires higher macro F1, equal or higher accuracy, and equal or lower loss.")
        st.button("Promote candidate", key=f"promote_{job['job_id']}",
                  disabled=not operator_ready or result["candidate_version"] == active, on_click=switch_model,
                  args=(f"/model/versions/{job['job_id']}/promote", active, "Candidate promoted."))


@st.fragment(run_every=3)
def follow_job(job_id):
    job = read_api(f"/train/jobs/{job_id}")
    if job is None:
        return
    if job["status"] in ("succeeded", "failed"):
        # Rebuild the page once to stop polling and enable another run.
        st.rerun()
    message = "Queued · waiting for a worker" if job["status"] == "queued" else STAGES.get(job["stage"], "Running")
    st.info(message)


st.title("Retraining")
operator_ready = st.session_state.get("operator_verified", False)
if not operator_ready:
    st.info("Enter your operator token in the sidebar to enable these controls.")
st.write("Fine-tune the serving model with a labeled batch and a sample of the original training photos.")
batches = read_api("/data/batches")
if batches is None:
    st.stop()
if not batches:
    st.info("Save a batch of labeled photos to start a training run.")
    st.page_link("bulk_upload.py", label="Open labeled uploads")
    st.stop()

by_id = {batch["batch_id"]: batch for batch in batches}


def batch_label(batch_id):
    batch = by_id[batch_id]
    noun = "image" if batch["image_count"] == 1 else "images"
    return f"{batch['crop'].capitalize()} · {batch['image_count']} {noun} · {batch_id}"


batch_id = st.selectbox("Saved batch", list(by_id), format_func=batch_label)
job_id = by_id[batch_id]["latest_job_id"]
job = read_api(f"/train/jobs/{job_id}") if job_id else None
pending = job is not None and job["status"] in ("queued", "running")
if st.button("Retrain", type="primary", disabled=not operator_ready or pending or bool(job_id and job is None)):
    try:
        response = httpx.post(api_url("/train/retrain"), headers=operator_headers(),
                              json={"batch_id": batch_id}, timeout=10)
        response.raise_for_status()
        # The batch listing now points to this job, including after a browser reload.
        st.rerun()
    except httpx.HTTPStatusError as error:
        st.error(error_detail(error.response))
    except httpx.RequestError:
        st.error("Could not confirm the training request. Refresh to check the latest job before retrying.")

notice = st.session_state.pop("model_switch_notice", None)
if notice:
    succeeded, message = notice
    if succeeded:
        st.success(message)
    else:
        st.error(message)
info = read_api("/model/info")
if info:
    st.caption("Serving model")
    st.code(info["model_version"], language=None)
    if info["model_version"] != info["baseline_version"]:
        st.button("Restore baseline", key="rollback", disabled=not operator_ready, on_click=switch_model,
                  args=("/model/rollback", info["model_version"], "Baseline restored."),
                  help=f"Switch predictions to {info['baseline_version']}.")
if job_id:
    st.subheader("Latest job")
    st.caption("Job ID")
    st.code(job_id, language=None)
    if pending:
        follow_job(job_id)
    elif job:
        show_result(job, info)

"""LeafVision dashboard entry point."""

import httpx
import streamlit as st

from frontend.api_client import api_url, error_detail

st.set_page_config(page_title="LeafVision", page_icon="🌱")
page = st.navigation([
    st.Page("crop_prediction.py", title="Crop prediction", default=True),
    st.Page("dataset_insights.py", title="Dataset insights", url_path="dataset-insights"),
    st.Page("bulk_upload.py", title="Labeled uploads", url_path="labeled-uploads"),
    st.Page("retraining.py", title="Retraining", url_path="retraining"),
    st.Page("monitoring.py", title="Monitoring", url_path="monitoring"),
], position="top")


def verify_operator():
    token = st.session_state.operator_token
    st.session_state.operator_verified = False
    st.session_state.operator_error = None
    if not token:
        return
    if not (token.isascii() and token.isprintable()) or " " in token:
        st.session_state.operator_error = "Enter a valid operator token"
        return
    try:
        response = httpx.get(api_url("/operator/access"), headers={"Authorization": f"Bearer {token}"}, timeout=10)
        response.raise_for_status()
        st.session_state.operator_verified = response.status_code == 204
    except httpx.HTTPStatusError as error:
        st.session_state.operator_error = error_detail(error.response)
    except httpx.RequestError:
        st.session_state.operator_error = "Could not verify operator access. Check the API and re-enter the token."


with st.sidebar:
    st.text_input("Operator token", type="password", key="operator_token", on_change=verify_operator,
                  help="Enter the token to manage training data and model versions. Clear it to sign out.")
    if st.session_state.get("operator_error"):
        st.error(st.session_state.operator_error)
    elif st.session_state.get("operator_verified"):
        st.caption("Operator access enabled")

page.run()

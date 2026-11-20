"""LeafVision dashboard entry point."""

import streamlit as st

st.set_page_config(page_title="LeafVision", page_icon="🌱")
page = st.navigation([
    st.Page("crop_prediction.py", title="Crop prediction", default=True),
    st.Page("dataset_insights.py", title="Dataset insights", url_path="dataset-insights"),
    st.Page("bulk_upload.py", title="Labeled uploads", url_path="labeled-uploads"),
    st.Page("retraining.py", title="Retraining", url_path="retraining"),
    st.Page("monitoring.py", title="Monitoring", url_path="monitoring"),
], position="top")
page.run()

"""Save crop-labeled JPEGs and display counts from durable upload storage."""

import httpx
import streamlit as st

from frontend.api_client import api_url, error_detail, operator_headers
from src import CROP_CLASSES
from src.schemas import UploadReceipt, UploadSummary


st.title("Labeled uploads")
operator_ready = st.session_state.get("operator_verified", False)
if not operator_ready:
    st.info("Enter your operator token in the sidebar to enable these controls.")
st.write("Choose one crop and upload new photos with that verified label. "
         "Keep evaluation photos separate for testing.")
crop = st.selectbox("Crop label", CROP_CLASSES, index=None, format_func=str.capitalize,
                    placeholder="Choose a crop", disabled=not operator_ready)
photos = st.file_uploader("Leaf photos", type=["jpg", "jpeg"], accept_multiple_files=True,
                          disabled=not operator_ready, max_upload_size=5,
                          help="Up to 10 JPEGs per batch, each up to 5 MiB and 20 million pixels.")
if len(photos) > 10:
    st.warning("Choose at most 10 photos for this batch.")
can_save = operator_ready and crop is not None and 1 <= len(photos) <= 10
if st.button("Save labeled batch", type="primary", disabled=not can_save):
    files = [("files", (photo.name, photo.getvalue(), "image/jpeg")) for photo in photos]
    try:
        with st.spinner("Saving photos…"):
            response = httpx.post(api_url("/data/upload"), headers=operator_headers(),
                                  data={"crop": crop}, files=files, timeout=60)
            response.raise_for_status()
            st.session_state.last_saved_batch = UploadReceipt.model_validate_json(response.content)
        st.success("Batch saved.")
    except httpx.HTTPStatusError as error:
        st.error(error_detail(error.response))
    except (httpx.RequestError, ValueError):
        st.error("Could not confirm whether the batch was saved. Check the upload counts before retrying.")

receipt = st.session_state.get("last_saved_batch")
if receipt:
    with st.container(border=True):
        st.subheader("Last saved batch")
        noun = "image" if receipt.image_count == 1 else "images"
        st.write(f"{receipt.crop.capitalize()} · {receipt.image_count} {noun}")
        st.caption("Batch ID")
        st.code(str(receipt.batch_id), language=None)

st.subheader("Saved uploads")
try:
    response = httpx.get(api_url("/data/summary"), timeout=10)
    response.raise_for_status()
    summary = UploadSummary.model_validate_json(response.content)
    counts = [{"Crop": label.capitalize(), "Images": summary.images_by_crop[label]} for label in CROP_CLASSES]
except httpx.HTTPStatusError as error:
    st.warning(f"Upload counts unavailable. {error_detail(error.response)}")
except (httpx.RequestError, ValueError, KeyError):
    st.warning("Could not read the saved upload counts. Refresh to try again.")
else:
    batches, images = st.columns(2)
    batches.metric("Batches", summary.total_batches)
    images.metric("Images", summary.total_images)
    st.table(counts, hide_index=True)
st.button("Refresh counts")

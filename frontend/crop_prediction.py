"""LeafVision's single-image crop prediction dashboard."""

import httpx
import streamlit as st

from frontend.api_client import api_url, error_detail
from src.schemas import Prediction


def clear_prediction():
    st.session_state.pop("prediction", None)


def classify(photo):
    response = httpx.post(api_url("/predict"), files={"file": (photo.name, photo.getvalue(), "image/jpeg")},
                          timeout=30)
    response.raise_for_status()
    return Prediction.model_validate_json(response.content)


st.title("LeafVision")
st.write("Identify cashew, cassava, maize, rice, or tomato from a close-up leaf photo.")

photo = st.file_uploader("Leaf photo", type=["jpg", "jpeg"], max_upload_size=5,
                        help="Choose one JPEG, up to 5 MiB and 20 million pixels.",
                        on_change=clear_prediction)
if st.button("Identify crop", type="primary", disabled=photo is None):
    clear_prediction()
    try:
        with st.spinner("Classifying the leaf…"):
            st.session_state.prediction = classify(photo)
    except httpx.HTTPStatusError as error:
        st.error(error_detail(error.response))
    except httpx.RequestError:
        st.error("Could not reach the prediction API. Check that it is running and try again.")
    except ValueError:
        st.error("The prediction API returned an unreadable result. Try again.")

result = st.session_state.get("prediction")
if photo is not None and result is not None:
    with st.container(border=True):
        preview, details = st.columns([1, 2], vertical_alignment="center")
        with preview:
            st.image(photo.getvalue(), width="stretch")
        with details:
            st.metric("Predicted crop", result.crop.capitalize())
            st.metric("Model score", f"{result.score:.8f}")
            st.caption("Serving model")
            st.text(result.model_version)

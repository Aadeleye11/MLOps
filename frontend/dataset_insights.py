"""Compare whole-image measurements across the five original training crops."""

import json
from pathlib import Path

import altair as alt
import streamlit as st

from src import CROP_CLASSES


def feature_chart(summary, feature, units):
    rows = [{"crop": crop.title(), "images": summary[crop]["images"], **summary[crop][feature]}
            for crop in CROP_CLASSES]
    chart = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("crop:N", title=None, sort=[row["crop"] for row in rows]),
        tooltip=[
            alt.Tooltip("crop:N", title="Crop"),
            alt.Tooltip("images:Q", title="Training images", format=",d"),
            alt.Tooltip("p10:Q", title="10th percentile", format=".2f"),
            alt.Tooltip("median:Q", title="Median", format=".2f"),
            alt.Tooltip("p90:Q", title="90th percentile", format=".2f"),
        ],
    )
    interval = chart.mark_rule(color="#83AA93", strokeWidth=5).encode(
        x=alt.X("p10:Q", title=units),
        x2="p90:Q",
    )
    median = chart.mark_point(filled=True, color="#175537", opacity=1, size=95).encode(
        x=alt.X("median:Q", title=units),
    )
    st.altair_chart((interval + median).properties(height=175), width="stretch")


summary_path = Path(__file__).with_name("data") / "training-summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
total = sum(summary[crop]["images"] for crop in CROP_CLASSES)
st.title("Dataset insights")
st.write(f"{total:,} original training images across five crops. Each unique JPEG byte sequence "
         "counts once. Features were measured across the whole frame at 128 × 128 pixels.")
st.caption("Dots mark medians. Lines span the 10th to 90th percentiles. Hover for values and image counts.")

rice = summary["rice"]
maize = summary["maize"]
st.subheader("Brightness")
feature_chart(summary, "brightness", "Mean grayscale intensity (0–255)")
other_medians = [summary[crop]["brightness"]["median"] for crop in CROP_CLASSES if crop != "rice"]
st.write(f"Rice had a median brightness of {rice['brightness']['median']:.1f} out of 255. "
         f"The other crops ranged from {min(other_medians):.1f} to {max(other_medians):.1f} "
         f"in median brightness. In the rice collection, the median share of near-white pixels "
         f"was {rice['near_white']['median']:.1f}%. The separate source and pale backgrounds "
         "could give the classifier a cue that changes in field photographs.")

st.subheader("Green dominance")
feature_chart(summary, "green_dominance", "Mean green minus the larger of red and blue")
st.write(f"Maize and cassava had similar median green dominance "
         f"({maize['green_dominance']['median']:+.1f} and "
         f"{summary['cassava']['green_dominance']['median']:+.1f}). Their ranges overlap, "
         f"while tomato had a median of {summary['tomato']['green_dominance']['median']:.1f}. "
         "Lighting and background color contribute to this measure, which limits its value "
         "as a crop identifier on its own.")

st.subheader("Edge density")
feature_chart(summary, "edge_density", "Interior pixels with a sharp brightness change (%)")
st.write(f"The middle 80% of maize images ranged from {maize['edge_density']['p10']:.2f}% "
         f"to {maize['edge_density']['p90']:.2f}% edge pixels. Rice had a narrower range "
         f"of {rice['edge_density']['p10']:.2f}% to {rice['edge_density']['p90']:.2f}%. "
         "A pixel counts if its brightness differs from its right or lower neighbor by at "
         "least 24 on a 0–255 scale. Uniform backgrounds can lower the result even when "
         "the leaf itself has visible texture.")

from __future__ import annotations

import plotly.express as px
import streamlit as st

try:
    from src.interface import make_prediction
except ImportError:
    from interface import make_prediction


st.set_page_config(page_title="AAPL Stock Outlook Assistant", page_icon="AAPL", layout="centered")

st.title("AAPL Stock Outlook Assistant")
st.write(
    "Ask a natural-language question about Apple's next-week stock outlook. "
    "The app parses your question, invokes the trained model, and shows a probability curve."
)

query = st.text_input(
    "Your question",
    value="What is the outlook for AAPL next week?",
    placeholder="Example: Will Apple move up over the next week?",
)

if st.button("Generate Outlook", type="primary"):
    with st.spinner("Running forecast..."):
        result = make_prediction(query)

    if result["status"] != "ok":
        st.warning(result["message"])
    else:
        up_probability = result["up_probability"]
        expected_return = result["expected_return"]
        uncertainty_std = result["uncertainty_std"]

        direction = "Higher" if result["prediction"] == 1 else "Not higher"
        st.subheader(f"Forecast: {direction}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probability Up", f"{up_probability:.1%}")
        col2.metric("Expected Return", f"{expected_return:.2%}")
        col3.metric("Typical Uncertainty", f"+/- {uncertainty_std:.2%}")

        st.write(result["message"])

        curve = result["curve"].copy()
        curve["Return (%)"] = curve["return"] * 100
        figure = px.line(
            curve,
            x="Return (%)",
            y="probability_density",
            title="Estimated 5-Day Return Probability Curve",
            labels={"probability_density": "Probability density"},
        )
        figure.add_vline(x=0, line_dash="dash", annotation_text="Break-even")
        figure.add_vline(x=expected_return * 100, line_dash="dot", annotation_text="Expected")
        st.plotly_chart(figure, use_container_width=True)

        with st.expander("Model details"):
            st.write(f"MLflow run ID: `{result.get('run_id')}`")
            st.json(result.get("metrics", {}))

st.caption("Educational project only. This is not financial advice.")

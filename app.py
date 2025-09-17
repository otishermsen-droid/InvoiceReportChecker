# app.py
import streamlit as st
import pandas as pd
from common import load_invoicing_report

st.set_page_config(page_title="Invoicing Sanity Checker", layout="wide")
st.title("Invoicing CSV Sanity Checker")

st.markdown(
    "Use the **pages** on the left: 1) Upload here, 2) Validate data, 3) Ask the assistant."
)

# Initialize session state holders once
if "df" not in st.session_state:
    st.session_state.df = None

# Upload
uploaded = st.file_uploader("Upload NAV Invoicing CSV", type=["csv"])

if uploaded:
    df = load_invoicing_report(uploaded)
    st.session_state.df = df  # persist across pages
    st.success(f"Loaded {len(df):,} rows.")
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Upload a CSV to begin.")

st.caption("Tip: Once uploaded, go to **Data Validation** in the left sidebar.")

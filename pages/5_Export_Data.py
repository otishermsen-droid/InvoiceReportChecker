import streamlit as st
import pandas as pd

st.set_page_config(page_title="Export Updated Data", layout="wide")
st.title("📤 Export Updated Data")

# Guard: need a dataframe in session
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Go to **File Upload** to load a CSV first.")
    st.stop()

df = st.session_state.df.copy()

st.markdown("## Preview Updated DataFrame")
st.dataframe(df.head(100))

st.markdown("---")
st.markdown("## Download")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download CSV",
    data=csv_bytes,
    file_name="updated_invoice_report.csv",
    mime="text/csv",
)



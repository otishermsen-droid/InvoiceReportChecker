import streamlit as st
import pandas as pd

st.set_page_config(page_title="Export Updated Data", layout="wide")
st.title("📤 Export Updated Data")

# Guard: need a dataframe in session
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Go to **File Upload** to load a CSV first.")
    st.stop()

df = st.session_state.df.copy()

# If brand HE, revert internal scaling by multiplying selected columns by 100
brand = st.session_state.get("brand")
if isinstance(brand, str) and brand.strip().upper() == "HE":
    he_cols = [
        "Qty", "% Tax", "Original Price", "Sell Price", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2", "COGS x Qty"
    ]
    for c in he_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0
    # Exchange rate back to original scale
    if "Exchange rate" in df.columns:
        df["Exchange rate"] = pd.to_numeric(df["Exchange rate"], errors="coerce") * 100000.0

    # Ensure no decimals for these columns in export
    for c in he_cols + ["Exchange rate"]:
        if c in df.columns:
            ser = pd.to_numeric(df[c], errors="coerce")
            df[c] = ser.round(0).astype("Int64")

# Remove internal calculation columns that shouldn't be in the final export
columns_to_remove = ["recalc_%TLG FEE"]
for col in columns_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])

st.markdown("## Preview Updated DataFrame")
st.dataframe(df.head(100))

st.markdown("---")
st.markdown("## Download")

# Export with EU-friendly formatting: semicolon separator and comma decimals
csv_text = df.to_csv(
    index=False,
    sep=";",
    decimal=",",
    na_rep="",
    float_format="%.2f",
)
csv_bytes = csv_text.encode("utf-8")
st.download_button(
    label="⬇️ Download CSV",
    data=csv_bytes,
    file_name="updated_invoice_report.csv",
    mime="text/csv",
)



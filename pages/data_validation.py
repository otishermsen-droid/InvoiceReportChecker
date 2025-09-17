# pages/1_📄_Data_Validation.py
import streamlit as st
import pandas as pd
from common import (
    sanity_check_cogs,
    sanity_checks_ddp_tax,
    sanity_check_gmv,
    fetch_orders_returns,
)

st.set_page_config(page_title="Raw File Data Validation", layout="wide")
st.title("File Data Validation")

# Require data
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No CSV loaded yet. Go back to **Home** and upload a file.")
    st.stop()

df = st.session_state.df

# Sidebar parameters
st.sidebar.header("Parameters")
coexist_tol = st.sidebar.number_input("DDP/%Tax tolerance", min_value=0.0, value=0.01, step=0.01)

with st.sidebar.expander("Optional BigQuery context (NOT used by AI)"):
    brand = st.text_input("Brand", value="SC")
    start_date = st.text_input("Start date (YYYY-MM-DD)", value="2025-08-01")
    end_date = st.text_input("End date (YYYY-MM-DD)", value="2025-08-31")
    fetch_bq = st.checkbox("Fetch BQ orders/returns for context", value=False)

# Run checks
cogs_mism = sanity_check_cogs(df.copy())
ddp_tax_mism = sanity_checks_ddp_tax(df.copy(), tol=coexist_tol)
gmv_mism = sanity_check_gmv(df.copy())

# Save for Assistant page
st.session_state.cogs_mism = cogs_mism
st.session_state.ddp_tax_mism = ddp_tax_mism
st.session_state.gmv_mism = gmv_mism
st.session_state.tols = {"coexist_tol": coexist_tol}

st.markdown("## Validation Results")

cogs_pass = len(cogs_mism) == 0
ddp_pass = len(ddp_tax_mism) == 0
gmv_pass = len(gmv_mism) == 0

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### COGS check")
    st.markdown(f"**{'PASSED ✅' if cogs_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(cogs_mism)}**")
    st.dataframe(cogs_mism.head(50), use_container_width=True)
    st.download_button(
        "Download COGS mismatches CSV",
        data=cogs_mism.to_csv(index=False).encode("utf-8-sig"),
        file_name="cogs_mismatches.csv",
        mime="text/csv",
    )

with col2:
    st.markdown("#### DDP/%Tax coexistence check")
    st.markdown(f"**{'PASSED ✅' if ddp_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(ddp_tax_mism)}**")
    st.dataframe(ddp_tax_mism.head(50), use_container_width=True)
    st.download_button(
        "Download DDP/Tax mismatches CSV",
        data=ddp_tax_mism.to_csv(index=False).encode("utf-8-sig"),
        file_name="ddp_tax_mismatches.csv",
        mime="text/csv",
    )

st.markdown("#### GMV check (Sell Price - Discount - DDP Services vs GMV EUR)")
st.markdown(f"**{'PASSED ✅' if gmv_pass else 'NOT PASSED ❌'}**")
st.write(f"Mismatches: **{len(gmv_mism)}**")
st.dataframe(gmv_mism.head(50), use_container_width=True)
st.download_button(
    "Download GMV mismatches CSV",
    data=gmv_mism.to_csv(index=False).encode("utf-8-sig"),
    file_name="gmv_mismatches.csv",
    mime="text/csv",
)

# Optional BQ context
if fetch_bq:
    with st.spinner("Fetching BQ data (context only)..."):
        try:
            bq_df = fetch_orders_returns(start_date, end_date, brand)
            st.success(f"Fetched {len(bq_df)} BQ rows (not exposed to AI).")
            st.dataframe(bq_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"BQ fetch failed: {e}")

# pages/1_📄_Data_Validation.py
import streamlit as st
import pandas as pd
from common import (
    sanity_check_cogs,
    sanity_checks_ddp_tax,
    sanity_check_gmv,
    fetch_orders_returns,
    auto_fix_cogs,
)

st.set_page_config(page_title="Raw File Data Validation", layout="wide")
st.title("File Data Validation")

# Require data
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No CSV loaded yet. Go back to **Home** and upload a file.")
    st.stop()

df = st.session_state.df

if "cogs_fix_feedback" not in st.session_state:
    st.session_state.cogs_fix_feedback = None

# Sidebar parameters
st.sidebar.header("Parameters")

with st.sidebar.expander("Optional BigQuery context (NOT used by AI)"):
    brand = st.text_input("Brand", value="SC")
    start_date = st.text_input("Start date (YYYY-MM-DD)", value="2025-08-01")
    end_date = st.text_input("End date (YYYY-MM-DD)", value="2025-08-31")
    fetch_bq = st.checkbox("Fetch BQ orders/returns for context", value=False)

# Run checks
cogs_mism = sanity_check_cogs(df.copy())
ddp_tax_mism = sanity_checks_ddp_tax(df.copy())
gmv_mism = sanity_check_gmv(df.copy())

# Save for Assistant page
st.session_state.cogs_mism = cogs_mism
st.session_state.ddp_tax_mism = ddp_tax_mism
st.session_state.gmv_mism = gmv_mism
st.session_state.tols = {}

st.markdown("## Validation Results")

cogs_pass = len(cogs_mism) == 0
ddp_pass = len(ddp_tax_mism) == 0
gmv_pass = len(gmv_mism) == 0

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### COGS check ")
    with st.expander("Show COGS calculation info"):
        st.info("""
**COGS Calculation:**
COGS is checked as:
`COGS ≈ GMV Net VAT - TLG Fee - DDP Services`
Rows are flagged if the calculated COGS does not match the expected value (within the removed tolerance).
""", icon="ℹ️")
    st.markdown(f"**{'PASSED ✅' if cogs_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(cogs_mism)}**")
    feedback = st.session_state.get("cogs_fix_feedback")
    if feedback is not None:
        rows_fixed = feedback.get("rows", 0)
        if feedback.get("status") == "success":
            st.success(
                f"Auto-fixed {rows_fixed} row{'s' if rows_fixed != 1 else ''}."
            )
        else:
            st.info("No COGS mismatches were available to fix.")
        st.session_state.cogs_fix_feedback = None
    cogs_cols = ["Order Number", "Product ID", "COGS", "GMV Net VAT", "TLG Fee", "DDP Services", "expected_cogs", "delta"]
    cogs_display = cogs_mism[[col for col in cogs_cols if col in cogs_mism.columns]]
    action_cols = st.columns(2)
    with action_cols[1]:
        fix_btn = st.button(
            "Auto-fix COGS mismatches",
            type="primary",
            disabled=cogs_pass,
            use_container_width=True,
        )
    if fix_btn:
        updated_df, rows_fixed = auto_fix_cogs(st.session_state.df, cogs_mism)
        st.session_state.df = updated_df
        st.session_state.cogs_fix_feedback = {
            "status": "success" if rows_fixed else "info",
            "rows": rows_fixed,
        }
        st.experimental_rerun()
    with action_cols[0]:
        st.download_button(
            "Download COGS mismatches CSV",
            data=cogs_display.to_csv(index=False).encode("utf-8-sig"),
            file_name="cogs_mismatches.csv",
            mime="text/csv",
        )
    st.dataframe(cogs_display.head(50), use_container_width=True)

with col2:
    st.markdown("#### DDP/%Tax coexistence check")
    with st.expander("Show DDP/%Tax calculation info"):
        st.info("""
**DDP/%Tax Coexistence Calculation:**  
Rows are flagged if both `% Tax` and `DDP Services` are greater than the specified tolerance.  
This checks for cases where both are present, which may indicate a data issue.
""", icon="ℹ️")
    st.markdown(f"**{'PASSED ✅' if ddp_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(ddp_tax_mism)}**")
    ddp_cols = ["Order Number", "Product ID", "% Tax", "DDP Services"]
    ddp_display = ddp_tax_mism[[col for col in ddp_cols if col in ddp_tax_mism.columns]]
    st.dataframe(ddp_display.head(50), use_container_width=True)
    st.download_button(
        "Download DDP/Tax mismatches CSV",
        data=ddp_display.to_csv(index=False).encode("utf-8-sig"),
        file_name="ddp_tax_mismatches.csv",
        mime="text/csv",
    )

st.markdown("#### GMV check (Sell Price - Discount - DDP Services vs GMV EUR)")
with st.expander("Show GMV calculation info"):
    st.info("""
**GMV Calculation:**  
GMV is checked as:  
`GMV EUR ≈ Sell Price - Discount - DDP Services`  
Rows are flagged if the calculated GMV does not match the expected value (within the removed tolerance).
""", icon="ℹ️")
st.markdown(f"**{'PASSED ✅' if gmv_pass else 'NOT PASSED ❌'}**")
st.write(f"Mismatches: **{len(gmv_mism)}**")
gmv_cols = ["Order Number", "Product ID", "Sell Price", "Discount", "DDP Services", "GMV EUR", "expected_gmv", "delta"]
gmv_display = gmv_mism[[col for col in gmv_cols if col in gmv_mism.columns]]
st.dataframe(gmv_display.head(50), use_container_width=True)
st.download_button(
    "Download GMV mismatches CSV",
    data=gmv_display.to_csv(index=False).encode("utf-8-sig"),
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

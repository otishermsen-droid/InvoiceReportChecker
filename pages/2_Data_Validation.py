# pages/1_📄_Data_Validation.py
import streamlit as st
import pandas as pd
from common import (
    sanity_check_cogs,
    sanity_checks_ddp_tax,
    sanity_check_gmv,
    fetch_orders_returns,
    auto_fix_cogs,
    sanity_check_tlg_fee,
    load_invoicing_report,
)

st.set_page_config(page_title="File Data Validation", layout="wide")
st.title("File Data Validation")

# Custom CSS for green auto-fix buttons
st.markdown("""
<style>
.stButton > button:first-child {
    background-color: #28a745;
    color: white;
    border: none;
}
.stButton > button:first-child:hover {
    background-color: #218838;
    color: white;
}
.stButton > button:first-child:disabled {
    background-color: #6c757d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Require data and persist uploaded file
# ────────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or st.session_state.df is None:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_invoicing_report(uploaded_file)
        st.session_state.df = df
        st.session_state.uploaded_file = uploaded_file.getvalue()
    else:
        st.warning("No CSV loaded yet. Go back to **Home** and upload a file.")
        st.stop()
else:
    # If file is not in memory but was uploaded before, reload it
    if "uploaded_file" in st.session_state and st.session_state.df is None:
        from io import BytesIO

        df = load_invoicing_report(BytesIO(st.session_state.uploaded_file))
        st.session_state.df = df
    else:
        df = st.session_state.df

if "cogs_fix_feedback" not in st.session_state:
    st.session_state.cogs_fix_feedback = None
if "tlg_fee_fix_feedback" not in st.session_state:
    st.session_state.tlg_fee_fix_feedback = None

# Check if we have updated TLG FEE mismatches in session state
if "tlg_fee_mism" in st.session_state:
    tlg_fee_mism = st.session_state.tlg_fee_mism
else:
    tlg_fee_mism = sanity_check_tlg_fee(df.copy(), atol=0.01, rtol=0.01)
    st.session_state.tlg_fee_mism = tlg_fee_mism

cogs_mism = sanity_check_cogs(df.copy(), atol=0.01, rtol=0.01)
ddp_tax_mism = sanity_checks_ddp_tax(df.copy())
gmv_mism = sanity_check_gmv(df.copy())

st.markdown("## Validation Results")

cogs_pass = len(cogs_mism) == 0
ddp_pass = len(ddp_tax_mism) == 0
gmv_pass = len(gmv_mism) == 0

col0, col1 = st.columns(2)

# ────────────────────────────────────────────────────────────────────────────────
# TLG FEE check (before COGS check)
# ────────────────────────────────────────────────────────────────────────────────
with col0:
    st.markdown("#### TLG FEE calculation check")
    with st.expander("Show TLG FEE calculation info"):
        st.info(
            """
**TLG FEE Calculation:**
TLG FEE is checked as:
`TLG Fee ≈ GMV Net VAT * (% TLG FEE / 100)`
Rows are flagged if the calculated TLG Fee does not match the expected value (within a small tolerance).
""",
            icon="ℹ️",
        )

    tlg_cols = [
        "Order Number",
        "Product ID",
        "TLG Fee",
        "GMV Net VAT",
        "% TLG FEE",
        "expected_tlg_fee",
        "delta",
    ]
    tlg_display = tlg_fee_mism[[
        c for c in tlg_cols if c in tlg_fee_mism.columns]]

    st.markdown(
        f"**{'PASSED ✅' if len(tlg_fee_mism) == 0 else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(tlg_fee_mism)}**")

    # Show previous fix feedback (if any)
    tlg_feedback = st.session_state.get("tlg_fee_fix_feedback")
    if tlg_feedback is not None:
        rows_fixed = tlg_feedback.get("rows", 0)
        if tlg_feedback.get("status") == "success":
            st.success(
                f"Auto-fixed {rows_fixed} row{'s' if rows_fixed != 1 else ''}.")
        elif tlg_feedback.get("status") == "info":
            st.info("No TLG FEE mismatches were available to fix.")
        elif tlg_feedback.get("status") == "warning":
            st.warning(tlg_feedback.get(
                "message", "Some rows could not be fixed."))
        st.session_state.tlg_fee_fix_feedback = None

    if not tlg_display.empty:
        st.dataframe(tlg_display.head(50), use_container_width=True)

    # Action button underneath dataframe
    fix_tlg_btn = st.button(
        "Auto-fix TLG FEE mismatches",
        type="primary",
        disabled=len(tlg_fee_mism) == 0,
        use_container_width=True,
    )

    print("tlg_fee_mism ->", tlg_fee_mism)

if fix_tlg_btn:
    try:
        # Simple approach: update TLG Fee with expected_tlg_fee for all mismatch rows
        updated_df = st.session_state.df.copy()

        # Get the indices of rows that have mismatches
        mismatch_indices = tlg_fee_mism.index

        # Update TLG Fee with expected values for mismatch rows
        rows_fixed = 0
        for idx in mismatch_indices:
            if idx in updated_df.index and 'expected_tlg_fee' in tlg_fee_mism.columns:
                expected_value = tlg_fee_mism.loc[idx, 'expected_tlg_fee']
                if pd.notna(expected_value):
                    updated_df.loc[idx, 'TLG Fee'] = expected_value
                    rows_fixed += 1

        # Update the main dataframe
        st.session_state.df = updated_df

        # Recalculate mismatches to update the validation page
        print("Debug - Before recalculation:")
        print(f"DataFrame shape: {st.session_state.df.shape}")
        print(
            f"TLG Fee sample values: {st.session_state.df['TLG Fee'].head(5).tolist()}")
        print(
            f"GMV Net VAT sample values: {st.session_state.df['GMV Net VAT'].head(5).tolist()}")
        print(
            f"% TLG FEE sample values: {st.session_state.df['% TLG FEE'].head(5).tolist()}")

        tlg_fee_mism = sanity_check_tlg_fee(
            st.session_state.df.copy(), atol=0.01, rtol=0.01)

        print("Debug - After recalculation:")
        print(f"New mismatches found: {len(tlg_fee_mism)}")
        if len(tlg_fee_mism) > 0:
            print(f"Sample mismatch data: {tlg_fee_mism.head(3)}")

        # Store updated mismatches in session state
        st.session_state.tlg_fee_mism = tlg_fee_mism

        st.session_state.tlg_fee_fix_feedback = {
            "status": "success",
            "rows": rows_fixed,
        }

    except Exception as e:
        st.session_state.tlg_fee_fix_feedback = {
            "status": "warning",
            "rows": 0,
            "message": f"TLG auto-fix failed: {e}",
        }

    st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# COGS check
# ────────────────────────────────────────────────────────────────────────────────
with col1:
    st.markdown("#### COGS check ")
    with st.expander("Show COGS calculation info"):
        st.info(
            """
**COGS Calculation:**
COGS is checked as:
`COGS ≈ GMV Net VAT - TLG Fee - DDP Services`
Rows are flagged if the calculated COGS does not match the expected value (within the removed tolerance).
""",
            icon="ℹ️",
        )
    st.markdown(f"**{'PASSED ✅' if cogs_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(cogs_mism)}**")

    feedback = st.session_state.get("cogs_fix_feedback")
    if feedback is not None:
        rows_fixed = feedback.get("rows", 0)
        if feedback.get("status") == "success":
            st.success(
                f"Auto-fixed {rows_fixed} row{'s' if rows_fixed != 1 else ''}.")
        else:
            st.info("No COGS mismatches were available to fix.")
        st.session_state.cogs_fix_feedback = None

    cogs_cols = [
        "Order Number",
        "Product ID",
        "COGS",
        "GMV Net VAT",
        "TLG Fee",
        "DDP Services",
        "expected_cogs",
        "delta",
    ]
    cogs_display = cogs_mism[[c for c in cogs_cols if c in cogs_mism.columns]]

    if not cogs_display.empty:
        st.dataframe(cogs_display.head(50), use_container_width=True)

    # Action button underneath dataframe
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
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# DDP/%Tax coexistence & GMV checks
# ────────────────────────────────────────────────────────────────────────────────
col2, col3 = st.columns(2)

with col2:
    st.markdown("#### DDP/%Tax coexistence check")
    with st.expander("Show DDP/%Tax calculation info"):
        st.info(
            """
**DDP/%Tax Coexistence Calculation:**  
Rows are flagged if both `% Tax` and `DDP Services` are greater than the specified tolerance.  
This checks for cases where both are present, which may indicate a data issue.
""",
            icon="ℹ️",
        )
    st.markdown(f"**{'PASSED ✅' if ddp_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(ddp_tax_mism)}**")
    ddp_cols = ["Order Number", "Product ID", "% Tax", "DDP Services"]
    ddp_display = ddp_tax_mism[[
        c for c in ddp_cols if c in ddp_tax_mism.columns]]

    if not ddp_display.empty:
        st.dataframe(ddp_display.head(50), use_container_width=True)

with col3:
    st.markdown(
        "#### GMV check (Sell Price - Discount - DDP Services vs GMV EUR)")
    with st.expander("Show GMV calculation info"):
        st.info(
            """
**GMV Calculation:**  
GMV is checked as:  
`GMV EUR ≈ Sell Price - Discount - DDP Services`  
Rows are flagged if the calculated GMV does not match the expected value (within the removed tolerance).
""",
            icon="ℹ️",
        )
    st.markdown(f"**{'PASSED ✅' if gmv_pass else 'NOT PASSED ❌'}**")
    st.write(f"Mismatches: **{len(gmv_mism)}**")
    gmv_cols = [
        "Order Number",
        "Product ID",
        "Sell Price",
        "Discount",
        "DDP Services",
        "GMV EUR",
        "expected_gmv",
        "delta",
    ]
    gmv_display = gmv_mism[[c for c in gmv_cols if c in gmv_mism.columns]]

    if not gmv_display.empty:
        st.dataframe(gmv_display.head(50), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Optional BQ context
# ────────────────────────────────────────────────────────────────────────────────
if 'fetch_bq' not in locals():
    fetch_bq = False

if fetch_bq:
    with st.spinner("Fetching BQ data (context only)..."):
        try:
            bq_df = fetch_orders_returns(start_date, end_date, brand)
            st.success(f"Fetched {len(bq_df)} BQ rows (not exposed to AI).")
            st.dataframe(bq_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"BQ fetch failed: {e}")

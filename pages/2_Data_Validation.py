# pages/1_📄_Data_Validation.py
import streamlit as st
import pandas as pd
from common import (
    sanity_check_cogs,
    sanity_checks_ddp_tax,
    sanity_check_gmv_eur,
    sanity_check_tlg_fee,
    load_invoicing_report,
    sanity_check_gmv_net
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

if "tlg_fee_fix_feedback" not in st.session_state:
    st.session_state.tlg_fee_fix_feedback = None
if "cogs_fix_feedback" not in st.session_state:
    st.session_state.cogs_fix_feedback = None

tlg_fee_mism = sanity_check_tlg_fee(df.copy(), atol=0.01, rtol=0.01)    
cogs_mism = sanity_check_cogs(df.copy(), atol=0.01, rtol=0.01)
ddp_tax_mism = sanity_checks_ddp_tax(df.copy())
gmv_eur_mism = sanity_check_gmv_eur(df.copy())
gmv_net_mism = sanity_check_gmv_net(df.copy())


st.markdown("## Validation Results")

cogs_pass = len(cogs_mism) == 0
ddp_pass = len(ddp_tax_mism) == 0
gmv_pass = len(gmv_eur_mism) == 0

# ────────────────────────────────────────────────────────────────────────────────
# DDP/%Tax coexistence 
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("#### 1. DDP/%Tax coexistence check")
st.markdown("This issue can't be auto-fixed. If there are any cases where both DDP and % Tax exist, please fix these manually in the csv and re-upload the new file to prevent reworks later on.")
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




# ────────────────────────────────────────────────────────────────────────────────
# GMV CHECK EUR
# ────────────────────────────────────────────────────────────────────────────────

st.markdown(
    "#### 2. GMV EUR Check")
with st.expander("Show GMV calculation info"):
    st.info(
        """
**GMV Calculation:**  
GMV is checked as:  
`GMV EUR ≈ Sell Price - Discount - DDP Services`  
Rows are flagged if the calculated GMV does not match the expected value.
""",
        icon="ℹ️",
    )
st.markdown(f"**{'PASSED ✅' if gmv_pass else 'NOT PASSED ❌'}**")
st.write(f"Mismatches: **{len(gmv_eur_mism)}**")
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


gmv_display = gmv_eur_mism[[c for c in gmv_cols if c in gmv_eur_mism.columns]]

# Display the GMV EUR mismatches table if not empty
if not gmv_display.empty:
    st.dataframe(gmv_display.head(50), use_container_width=True)

# Feedback banner (optional)
if "gmv_eur_fix_feedback" not in st.session_state:
    st.session_state.gmv_eur_fix_feedback = None
eu_feedback = st.session_state.get("gmv_eur_fix_feedback")
if eu_feedback is not None:
    rows_fixed = eu_feedback.get("rows", 0)
    status = eu_feedback.get("status")
    if status == "success":
        st.success(f"Auto-fixed {rows_fixed} row{'s' if rows_fixed != 1 else ''}.")
    elif status == "info":
        st.info("No GMV EUR mismatches were available to fix.")
    elif status == "warning":
        st.warning(eu_feedback.get("message", "Some rows could not be fixed."))
    st.session_state.gmv_eur_fix_feedback = None

# Auto-fix button
fix_gmv_eur_btn = st.button(
    "Auto-fix GMV EUR mismatches",
    type="primary",
    disabled=len(gmv_eur_mism) == 0,
    use_container_width=True,
)

if fix_gmv_eur_btn:
    try:
        updated_df = st.session_state.df.copy()

        # vectorized: align on index intersection, ignore NaNs in expected
        idx = gmv_eur_mism.index.intersection(updated_df.index)
        expected = gmv_eur_mism.loc[idx, "expected_gmv"]
        mask = expected.notna()
        updated_df.loc[idx[mask], "GMV EUR"] = expected[mask].values
        rows_fixed = int(mask.sum())

        st.session_state.df = updated_df

        # Recompute *all* affected checks (GMV EUR influences GMV Net VAT & downstream COGS)
        gmv_eur_mism = sanity_check_gmv_eur(st.session_state.df.copy())
        gmv_net_mism = sanity_check_gmv_net(st.session_state.df.copy())
        cogs_mism = sanity_check_cogs(st.session_state.df.copy(), atol=0.01, rtol=0.01)

        st.session_state.gmv_eur_fix_feedback = {
            "status": "success" if rows_fixed else "info",
            "rows": rows_fixed,
        }
    except Exception as e:
        st.session_state.gmv_eur_fix_feedback = {
            "status": "warning",
            "rows": 0,
            "message": f"GMV EUR auto-fix failed: {e}",
        }
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────────
# GMV CHECK NET VAT
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("#### 3. GMV Net VAT Check")
with st.expander("Show GMV Net VAT calculation info"):
    st.info(
        """
**GMV Net VAT Calculation:**  
GMV Net VAT is checked as:  
`GMV Net VAT ≈ GMV EUR / (1 + (% Tax / 100))`  
Rows are flagged if the calculated GMV Net VAT does not match the expected value.
""",
        icon="ℹ️",
    )
gmv_net_pass = len(gmv_net_mism) == 0
st.markdown(f"**{'PASSED ✅' if gmv_net_pass else 'NOT PASSED ❌'}**")
st.write(f"Mismatches: **{len(gmv_net_mism)}**")
gmv_net_cols = [
    "Order Number",
    "Product ID",
    "GMV EUR",
    "% Tax",
    "GMV Net VAT",
    "expected_gmv_net",
    "delta",
]
gmv_net_display = gmv_net_mism[[c for c in gmv_net_cols if c in gmv_net_mism.columns]]

# Display the GMV EUR mismatches table if not empty
if not gmv_net_display.empty:
    st.dataframe(gmv_net_display.head(50), use_container_width=True)

# Feedback banner (optional)
if "gmv_net_fix_feedback" not in st.session_state:
    st.session_state.gmv_net_fix_feedback = None
net_feedback = st.session_state.get("gmv_net_fix_feedback")
if net_feedback is not None:
    rows_fixed = net_feedback.get("rows", 0)
    status = net_feedback.get("status")
    if status == "success":
        st.success(f"Auto-fixed {rows_fixed} row{'s' if rows_fixed != 1 else ''}.")
    elif status == "info":
        st.info("No GMV Net VAT mismatches were available to fix.")
    elif status == "warning":
        st.warning(net_feedback.get("message", "Some rows could not be fixed."))
    st.session_state.gmv_net_fix_feedback = None

# Auto-fix button
fix_gmv_net_btn = st.button(
    "Auto-fix GMV Net VAT mismatches",
    type="primary",
    disabled=len(gmv_net_mism) == 0,
    use_container_width=True,
)

if fix_gmv_net_btn:
    try:
        updated_df = st.session_state.df.copy()

        # vectorized: align on index intersection, ignore NaNs in expected
        idx = gmv_net_mism.index.intersection(updated_df.index)
        expected = gmv_net_mism.loc[idx, "expected_gmv_net"]
        mask = expected.notna()
        updated_df.loc[idx[mask], "GMV Net VAT"] = expected[mask].values
        rows_fixed = int(mask.sum())

        st.session_state.df = updated_df

        # Recompute affected checks (GMV Net VAT impacts COGS)
        gmv_net_mism = sanity_check_gmv_net(st.session_state.df.copy())
        cogs_mism = sanity_check_cogs(st.session_state.df.copy(), atol=0.01, rtol=0.01)

        st.session_state.gmv_net_fix_feedback = {
            "status": "success" if rows_fixed else "info",
            "rows": rows_fixed,
        }
    except Exception as e:
        st.session_state.gmv_net_fix_feedback = {
            "status": "warning",
            "rows": 0,
            "message": f"GMV Net VAT auto-fix failed: {e}",
        }
    st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# 1 - TLG FEE check 
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("#### 4. TLG FEE calculation check")
with st.expander("Show TLG FEE calculation info"):
    st.info(
        """
**TLG FEE Calculation:**
TLG FEE is checked as:
`TLG Fee ≈ GMV Net VAT * (% TLG FEE / 100)`
Rows are flagged if the calculated TLG Fee does not match the expected value.
""",
        icon="ℹ️",
    )

tlg_cols = [
    "Order Number",
    "Product ID",
    "GMV Net VAT",
    "% TLG FEE",
"TLG Fee",
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

        tlg_fee_mism = sanity_check_tlg_fee(
            st.session_state.df.copy(), atol=0.01, rtol=0.01)

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
# 5 - COGS check
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("#### 5. COGS check ")
with st.expander("Show COGS calculation info"):
    st.info(
        """
**COGS Calculation:**
COGS is checked as:
`COGS ≈ GMV Net VAT - TLG Fee`
Rows are flagged if the calculated COGS does not match the expected value.
""",
        icon="ℹ️",
    )

st.markdown(f"**{'PASSED ✅' if cogs_pass else 'NOT PASSED ❌'}**")
st.write(f"Mismatches: **{len(cogs_mism)}**")

# Show previous fix feedback (if any)
feedback = st.session_state.get("cogs_fix_feedback")
if feedback is not None:
    rows_fixed = feedback.get("rows", 0)
    status = feedback.get("status")
    if status == "success":
        st.success(f"Auto-fixed {rows_fixed} row{'s' if rows_fixed != 1 else ''}.")
    elif status == "info":
        st.info("No COGS mismatches were available to fix.")
    elif status == "warning":
        st.warning(feedback.get("message", "Some rows could not be fixed."))
    st.session_state.cogs_fix_feedback = None

cogs_cols = [
    "Order Number",
    "Product ID",
    "GMV Net VAT",
    "TLG Fee",
    "DDP Services",
    "COGS",
    "expected_cogs",
    "delta",
]
cogs_display = cogs_mism[[c for c in cogs_cols if c in cogs_mism.columns]]

if not cogs_display.empty:
    st.dataframe(cogs_display.head(50), use_container_width=True)

# Action button underneath dataframe
fix_cogs_btn = st.button(
    "Auto-fix COGS mismatches",
    type="primary",
    disabled=len(cogs_mism) == 0,
    use_container_width=True,
)

if fix_cogs_btn:
    try:
        updated_df = st.session_state.df.copy()

        # Indices for mismatch rows
        mismatch_indices = cogs_mism.index

        rows_fixed = 0
        for idx in mismatch_indices:
            if idx in updated_df.index and "expected_cogs" in cogs_mism.columns:
                expected_value = cogs_mism.loc[idx, "expected_cogs"]
                if pd.notna(expected_value):
                    updated_df.loc[idx, "COGS"] = expected_value
                    if "COGS2" in updated_df.columns:
                        updated_df.loc[idx, "COGS2"] = expected_value
                    rows_fixed += 1

        # Update the main dataframe
        st.session_state.df = updated_df

        # Recalculate mismatches and cache
        cogs_mism = sanity_check_cogs(st.session_state.df.copy(), atol=0.01, rtol=0.01)
        st.session_state.cogs_mism = cogs_mism

        # Feedback
        st.session_state.cogs_fix_feedback = {
            "status": "success" if rows_fixed else "info",
            "rows": rows_fixed,
        }

    except Exception as e:
        st.session_state.cogs_fix_feedback = {
            "status": "warning",
            "rows": 0,
            "message": f"COGS auto-fix failed: {e}",
        }

    st.rerun()
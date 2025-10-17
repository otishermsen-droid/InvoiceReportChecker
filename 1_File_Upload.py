import streamlit as st
import pandas as pd
from io import BytesIO

from common import (
    load_invoicing_report,
    get_supported_brands,
    fetch_tlg_fee_config,
    fetch_ytd_totals_for_brand,
    decide_fee_percent_from_config,
    fetch_ytd_totals_until_date,
    apply_tlg_fee_config_per_row,
)

st.set_page_config(page_title="File Upload", layout="wide")
st.title("📥 File Upload")

# Sidebar settings to mirror other pages
with st.sidebar:
    st.markdown("### Settings")
    brand_options = get_supported_brands()
    brand_default_idx = brand_options.index("PJ") if "PJ" in brand_options else 0
    preselected = st.session_state.get("brand")
    if preselected in brand_options:
        brand_default_idx = brand_options.index(preselected)
    brand = st.selectbox(
        "Brand (2-letter code)",
        options=brand_options,
        index=brand_default_idx,
        help="Pick the brand whose column mapping should be applied",
    )
    st.session_state["brand"] = brand

    erpEntity_options = ["THE LEVEL", "TLG_USA", "TLG_UK"]
    erp_default_idx = 0
    prior_erp = st.session_state.get("erpEntity")
    if prior_erp in erpEntity_options:
        erp_default_idx = erpEntity_options.index(prior_erp)
    erpEntity = st.selectbox(
        "ERP Entity",
        options=erpEntity_options,
        index=erp_default_idx,
        help="Select the ERP Entity",
    )
    st.session_state["erpEntity"] = erpEntity

    st.subheader("📅 Date Range Context")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", help="Used to seed YTD before your file period")
    with c2:
        end_date = st.date_input("End Date", help="For display context only")
    st.session_state.validation_start_date = start_date
    st.session_state.validation_end_date = end_date

st.markdown("## Upload CSV")

uploaded_file = st.file_uploader("Upload the original CSV file", type=["csv"], help="Semicolon separated, quoted values")

if uploaded_file is not None and brand:
    try:
        # Load CSV according to brand/ERP mapping
        df = load_invoicing_report(uploaded_file, brand=brand, erp_entity=erpEntity)

        # Ensure state maps exist
        if "tlg_fee_config" not in st.session_state:
            st.session_state.tlg_fee_config = {}
        if "ytd_totals" not in st.session_state:
            st.session_state.ytd_totals = {}
        if "applied_tlg_fee_percent" not in st.session_state:
            st.session_state.applied_tlg_fee_percent = {}

        # Fetch brand config and YTD totals caches
        if brand not in st.session_state.tlg_fee_config:
            try:
                cfg_df = fetch_tlg_fee_config(brand)
            except Exception:
                cfg_df = pd.DataFrame()
            st.session_state.tlg_fee_config[brand] = cfg_df
        else:
            cfg_df = st.session_state.tlg_fee_config[brand]

        if brand not in st.session_state.ytd_totals:
            try:
                ytd_totals_df = fetch_ytd_totals_for_brand(brand)
            except Exception:
                ytd_totals_df = pd.DataFrame()
            st.session_state.ytd_totals[brand] = ytd_totals_df
        else:
            ytd_totals_df = st.session_state.ytd_totals[brand]

        # Seed running totals until chosen start date
        initial_gmv = 0.0
        initial_nmv = 0.0
        try:
            if st.session_state.validation_start_date is not None:
                ytd_until = fetch_ytd_totals_until_date(
                    brand=brand,
                    end_date=st.session_state.validation_start_date.strftime("%Y-%m-%d"),
                )
                if ytd_until is not None and not ytd_until.empty:
                    last = ytd_until.iloc[-1]
                    initial_gmv = float(pd.to_numeric(last.get("total_gmv_eur"), errors="coerce") or 0.0)
                    initial_nmv = float(pd.to_numeric(last.get("total_nmv_eur"), errors="coerce") or 0.0)
        except Exception:
            pass

        # Compute per-row recalc percent using static YTD totals and config filters
        try:
            df = apply_tlg_fee_config_per_row(
                df,
                brand=brand,
                fee_config_df=cfg_df,
                erp_entity=erpEntity,
                static_ytd_gmv=initial_gmv,
                static_ytd_nmv=initial_nmv,
            )
        except Exception:
            pass

        # Persist state
        st.session_state.df = df
        st.session_state.uploaded_file = uploaded_file.getvalue()

        st.success("✅ File loaded. You can now go to Data Validation to perform sanity checks.")

        st.markdown("## Preview")
        st.dataframe(df.head(50))

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
else:
    st.info("Select a brand and upload a CSV to begin.")



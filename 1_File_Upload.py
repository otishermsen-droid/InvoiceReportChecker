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
    fetch_reset_aware_totals,
    apply_tlg_fee_config_per_row,
    fetch_orders_returns,
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

        # Seed running totals until chosen start date, reset-aware
        initial_gmv = 0.0
        initial_nmv = 0.0
        try:
            if st.session_state.validation_start_date is not None:
                start_str = st.session_state.validation_start_date.strftime("%Y-%m-%d")
                end_str = st.session_state.validation_end_date.strftime("%Y-%m-%d") if st.session_state.validation_end_date is not None else start_str
                gmv_val, nmv_val = fetch_reset_aware_totals(
                    brand=brand,
                    period_start_date=start_str,
                    period_end_date=end_str,
                )
                initial_gmv = float(gmv_val or 0.0)
                initial_nmv = float(nmv_val or 0.0)
                print(initial_gmv, initial_nmv)
                # Persist for downstream display
                st.session_state.initial_gmv = initial_gmv
                st.session_state.initial_nmv = initial_nmv
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
            print(df["recalc_%TLG FEE"].notna().sum(), "non-null fees")
            print(df[["recalc_%TLG FEE"]].head())
        except Exception:
            pass

        # FE-only: enrich Gross price from BI on upload time
        try:
            if isinstance(brand, str) and brand.strip().upper() == "FE":
                start_dt = st.session_state.get("validation_start_date")
                end_dt = st.session_state.get("validation_end_date")
                if start_dt and end_dt and erpEntity:
                    start_str = start_dt.strftime("%Y-%m-%d")
                    end_str = end_dt.strftime("%Y-%m-%d")
                    # Build unique order_numbers from CSV to fetch outside timeframe if needed
                    order_numbers = []
                    if "Order Number" in df.columns:
                        order_numbers = (
                            df["Order Number"].astype(str).str.strip().str.upper().dropna().unique().tolist()
                        )
                    bq_df = fetch_orders_returns(start_str, end_str, brand, erpEntity, order_numbers=order_numbers)
                    if bq_df is not None and not bq_df.empty:
                        if "channel" in bq_df.columns:
                            bq_df = bq_df[bq_df["channel"].astype(str).str.strip().str.upper() == "B2C"]
                        # Always take gross_price from Orders rows (row_type = 'O')
                        if "row_type" in bq_df.columns:
                            bq_df = bq_df[bq_df["row_type"].astype(str).str.strip().str.upper() == "O"]
                        # Build keys (include row_type mapping: CSV Type S->O, R->R)
                        if "Order Number" in df.columns:
                            df["_order_key"] = df["Order Number"].astype(str).str.strip().str.upper()
                        else:
                            df["_order_key"] = pd.Series([None] * len(df), index=df.index)
                        if "EAN" in df.columns:
                            df["_ean_key"] = df["EAN"].astype(str).str.strip().str.upper()
                        else:
                            df["_ean_key"] = pd.Series([None] * len(df), index=df.index)
                        bq_df["_order_key"] = bq_df["order_number"].astype(str).str.strip().str.upper() if "order_number" in bq_df.columns else pd.Series([], dtype=str)
                        if "product_id" in bq_df.columns:
                            bq_df["_ean_key"] = bq_df["product_id"].astype(str).str.strip().str.upper().str.replace(r"^48", "", regex=True)
                        else:
                            bq_df["_ean_key"] = pd.Series([], dtype=str)

                        if "gross_price" in bq_df.columns:
                            bq_small = (
                                bq_df[["_order_key", "_ean_key", "gross_price", "discount"]]
                                .dropna(subset=["_order_key", "_ean_key"])
                                .sort_values(["_order_key", "_ean_key"])
                                .drop_duplicates(subset=["_order_key", "_ean_key"], keep="first")
                            )
                            df = df.merge(
                                bq_small,
                                how="left",
                                left_on=["_order_key", "_ean_key"],
                                right_on=["_order_key", "_ean_key"],
                                suffixes=("", "_bq"),
                            )
                            df["Gross price"] = pd.to_numeric(df.get("gross_price"), errors="coerce")
                            if "discount" in df.columns:
                                df["discount"] = pd.to_numeric(df.get("discount"), errors="coerce")
                            df = df.drop(columns=[c for c in ["_order_key", "_ean_key", "gross_price"] if c in df.columns])
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



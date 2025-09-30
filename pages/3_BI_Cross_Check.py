# pages/4_Create_Reports.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from common import fetch_orders_returns
import numpy as np
import pandas as pd


st.set_page_config(page_title="BQ Cross Check", layout="wide")
st.title("📊 BQ Cross Check")

# Check if data is available
if "df" not in st.session_state or st.session_state.df is None:
    st.warning(
        "No data loaded. Please go to **Data Validation** and upload a CSV file first.")
    st.stop()
else:
    csv_df = st.session_state.df.copy()


# Sidebar for BigQuery configuration (header only, settings removed)
st.sidebar.header("🔧 BigQuery Configuration")

# Brand input
brand = st.sidebar.text_input(
    "Brand (2-letter acronym)",
    value="PJ",
    help="Enter a 2-letter brand code (e.g. DG, SC)",
    max_chars=2
).upper()

erpEntity = st.sidebar.selectbox(
    "ERP Entity",
    options=["THE LEVEL", "TLG_USA", "TLG_UK"],
    index=0,
    help="Select the ERP Entity",
)

# Date range inputs
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=30),
        help="Select the start date for the query"
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        help="Select the end date for the query"
    )

# Validate date range
if start_date > end_date:
    st.sidebar.error("Start date must be before end date!")
    st.stop()

# Convert dates to string format DD-MM-YYYY
start_date_str = start_date.strftime("%d-%m-%Y")
end_date_str = end_date.strftime("%d-%m-%Y")

# Main content area
st.markdown("## BigQuery Data Fetch")

# Query button
if st.button("🔍 Query BigQuery", type="primary", width='stretch'):
    if not brand or len(brand) != 2:
        st.error("Please enter a valid 2-letter brand code!")
    else:
        with st.spinner("Fetching data from BigQuery..."):
            try:
                # Fetch data from BigQuery
                bq_df = fetch_orders_returns(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    brand=brand,
                    source_company=erpEntity
                )

                if bq_df.empty:
                    st.warning("No data found for the specified criteria.")
                else:
                    # Store in session state
                    st.session_state.bq_df = bq_df
                    st.session_state.bq_params = {
                        "brand": brand,
                        "start_date": start_date_str,
                        "end_date": end_date_str
                    }

                    st.success(
                        f"✅ Successfully fetched {len(bq_df):,} rows from BigQuery!")

            except Exception as e:
                st.error(f"❌ Error fetching data from BigQuery: {str(e)}")
                st.info("Please check your BigQuery configuration and try again.")

    # Display results if available
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("CSV not loaded yet. Go to **Data Validation** to upload it, then return here.")
    else:
        csv_df = st.session_state.df.copy()
        bq_df = st.session_state.bq_df.copy() if "bq_df" in st.session_state else None
        if bq_df is None or bq_df.empty:
            st.warning("No BigQuery data available yet. Use **Query BigQuery** above.")

    st.markdown("---")
    st.markdown("## 📋 Query Results")

    # Show the actual data
    st.markdown("#### Preview Data")

    # Display the dataframe
    st.dataframe(
        bq_df.head(50),
        width='stretch',
        height=400
    )

    # Data analysis
    st.markdown("### 👀 BQ Cross Check")

    st.markdown("#### Unique orders")

    csv_orders = csv_df[csv_df["Type"] == "Order"]

    bq_orders = bq_df[(bq_df["quantity"] > 0) & (bq_df["channel"] == "B2C")]

    # Normalize order IDs for robust comparison
    def _norm_series(s):
        return (
            s.astype(str)
             .str.strip()
             .str.upper()
        )

    csv_order_ids = _norm_series(csv_orders["Order Number"]).dropna()
    bq_order_ids  = _norm_series(bq_orders["order_number"]).dropna()

    csv_only_ids = sorted(set(csv_order_ids) - set(bq_order_ids))
    bq_only_ids  = sorted(set(bq_order_ids)  - set(csv_order_ids))

    # Filter original dataframes to show rows for the missing IDs (deduped per order)
    csv_only_tbl = (
        csv_orders[csv_orders["Order Number"].str.strip().str.upper().isin(csv_only_ids)]
        .drop_duplicates(subset=["Order Number"])
        .sort_values("Order Number")
        .reset_index(drop=True)
    )
    bq_only_tbl = (
        bq_orders[bq_orders["order_number"].str.strip().str.upper().isin(bq_only_ids)]
        .drop_duplicates(subset=["order_number"])
        .sort_values("order_number")
        .reset_index(drop=True)
    )

    col_left, col_right = st.columns(2)

    with col_left:
        count = csv_orders.drop_duplicates(subset=["Order Number"]).shape[0]
        st.markdown("**CSV Orders**")
        st.caption(f"Total orders: {count:,}")
        st.caption(f"Orders not in BQ: {len(csv_only_ids):,} order(s)")
        st.dataframe(
            csv_only_tbl,
            width='stretch',
            height=300
        )

    with col_right:
        count = bq_orders.drop_duplicates(subset=["order_number"]).shape[0]
        st.markdown("**BigQuery Orders**")
        st.caption(f"Total orders: {count:,}")
        st.caption(f"Orders not in CSV: {len(bq_only_ids):,} order(s)")
        st.dataframe(
            bq_only_tbl,
            width='stretch',
            height=300
        )

        # --- Units sold comparison (per PRODUCT) ---
    # --- Units sold comparison (per PRODUCT) ---
    st.markdown("#### Units sold")

    # Check required columns exist
    if "Product ID" not in csv_orders.columns:
        st.error('CSV is missing "Product ID" column.')
    elif "product_id" not in bq_orders.columns:
        st.error('BigQuery data is missing "product_id" column.')
    else:
        # Normalize product keys
        csv_orders["_prod_key"] = csv_orders["Product ID"].astype(str).str.strip().str.upper()
        bq_orders["_prod_key"]  = bq_orders["product_id"].astype(str).str.strip().str.upper()

        # Count units per product (quantity == 1 per row)
        csv_units_by_prod = (
            csv_orders.groupby("_prod_key")
            .size()
            .rename("csv_units")
        )
        bq_units_by_prod = (
            bq_orders.groupby("_prod_key")
            .size()
            .rename("bq_units")
        )

        # Combine into comparison table
        prod_compare = pd.concat([csv_units_by_prod, bq_units_by_prod], axis=1)

        # Totals
        csv_total_units = int(csv_units_by_prod.sum())
        bq_total_units  = int(bq_units_by_prod.sum())
        delta_units     = bq_total_units - csv_total_units

        st.caption(
            f"**Totals (units)** — CSV: {csv_total_units:,} | "
            f"BQ: {bq_total_units:,} | Δ (BQ−CSV): {delta_units:+,}"
        )

        # Masks
        in_both_mask   = prod_compare["csv_units"].notna() & prod_compare["bq_units"].notna()
        mismatch_mask  = in_both_mask & (prod_compare["csv_units"] != prod_compare["bq_units"])
        csv_only_mask  = prod_compare["csv_units"].notna() & prod_compare["bq_units"].isna()
        bq_only_mask   = prod_compare["csv_units"].isna() & prod_compare["bq_units"].notna()

        # Tables
        mismatches_tbl = (
            prod_compare.loc[mismatch_mask]
            .assign(diff=lambda d: d["bq_units"] - d["csv_units"])
            .reset_index(names=["product_id"])
            .sort_values(["diff", "product_id"], ascending=[False, True])
            .reset_index(drop=True)
        )

        csv_only_units_tbl = (
            prod_compare.loc[csv_only_mask, ["csv_units"]]
            .reset_index(names=["product_id"])
            .sort_values("product_id")
            .reset_index(drop=True)
        )

        bq_only_units_tbl = (
            prod_compare.loc[bq_only_mask, ["bq_units"]]
            .reset_index(names=["product_id"])
            .sort_values("product_id")
            .reset_index(drop=True)
        )

        # Display in 3 columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Mismatched units (by product)**")
            st.caption(f"{len(mismatches_tbl):,} product(s)")
            st.dataframe(mismatches_tbl, width='stretch', height=300)

        with col2:
            st.markdown("**Only in CSV (by product)**")
            st.caption(f"{len(csv_only_units_tbl):,} product(s)")
            st.dataframe(csv_only_units_tbl, width='stretch', height=300)

        with col3:
            st.markdown("**Only in BQ (by product)**")
            st.caption(f"{len(bq_only_units_tbl):,} product(s)")
            st.dataframe(bq_only_units_tbl, width='stretch', height=300)


    st.markdown("#### Total returns")
        # --- Returns comparison (same style as Unique orders) ---

    # Build returns subsets (per your definitions)
    csv_returns = csv_df[csv_df["Type"] == "Return"]
    bq_returns  = bq_df[(bq_df["returned_quantity"] > 0) & (bq_df["channel"] == "B2C")]

    # Safety checks
    if "Order Number" not in csv_returns.columns:
        st.error('CSV returns are missing "Order Number" column.')
    elif "order_number" not in bq_returns.columns:
        st.error('BigQuery returns are missing "order_number" column.')
    else:
        # Normalize order IDs
        def _norm_series(s):
            return s.astype(str).str.strip().str.upper()

        csv_ret_ids = _norm_series(csv_returns["Order Number"]).dropna()
        bq_ret_ids  = _norm_series(bq_returns["order_number"]).dropna()

        # Set differences
        csv_only_ret_ids = sorted(set(csv_ret_ids) - set(bq_ret_ids))
        bq_only_ret_ids  = sorted(set(bq_ret_ids)  - set(csv_ret_ids))

        # Deduped preview tables (one row per order)
        csv_only_ret_tbl = (
            csv_returns[csv_returns["Order Number"].astype(str).str.strip().str.upper().isin(csv_only_ret_ids)]
            .drop_duplicates(subset=["Order Number"])
            .sort_values("Order Number")
            .reset_index(drop=True)
        )
        bq_only_ret_tbl = (
            bq_returns[bq_returns["order_number"].astype(str).str.strip().str.upper().isin(bq_only_ret_ids)]
            .drop_duplicates(subset=["order_number"])
            .sort_values("order_number")
            .reset_index(drop=True)
        )

        # Counts
        csv_unique_returns = csv_returns.drop_duplicates(subset=["Order Number"]).shape[0]
        bq_unique_returns  = bq_returns.drop_duplicates(subset=["order_number"]).shape[0]

        # Display side-by-side
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**CSV returns**")
            st.caption(f"Total unique returns: {csv_unique_returns:,}")
            st.caption(f"Returns not in BQ: {len(csv_only_ret_ids):,} order(s)")
            st.dataframe(csv_only_ret_tbl, width='stretch', height=300)

        with col_r:
            st.markdown("**BigQuery returns**")
            st.caption(f"Total unique returns: {bq_unique_returns:,}")
            st.caption(f"Returns not in CSV: {len(bq_only_ret_ids):,} order(s)")
            st.dataframe(bq_only_ret_tbl, width='stretch', height=300)
  

st.markdown("### 🗽 US Duties Check")

ddp_col1, ddp_col2 = st.columns([1, 3])
with ddp_col1:
    ddp_clicked = st.button("🔎 Fetch DDP Config & Origins", width='stretch')

if ddp_clicked:
    try:
        with st.spinner("Resolving brand code and fetching DDP configuration..."):
            from common import fetch_brand_code, fetch_ddp_config, fetch_items_origin

            # 1) Resolve brand_code from config.brand
            brand_code = fetch_brand_code(brand) or brand  # fallback to input if not found
            st.session_state.ddp_brand_code = brand_code

            # 2) Fetch DDP config for brand_code + ERP Entity (source_company)
            ddp_cfg = fetch_ddp_config(brand_code=brand_code, source_company=erpEntity)
            if ddp_cfg.empty:
                st.warning("No DDP config rows found for this Brand/ERP Entity.")
            st.session_state.ddp_config = ddp_cfg

            # 3) Collect unique product_ids from CSV orders (US only)
            csv_orders_us = csv_df[
                (csv_df["Type"] == "Order") &
                (csv_df["Shipping Country"].astype(str).str.upper() == "US")
            ].copy()

            if csv_orders_us.empty:
                st.warning("No US orders found in the CSV for DDP check.")
                st.stop()

            candidate_cols = [c for c in ["Product ID", "ProductID", "product_id"] if c in csv_orders_us.columns]
            if not candidate_cols:
                st.error("CSV is missing a product id column (expected one of: Product ID, ProductID, product_id).")
                st.stop()

            prod_col = candidate_cols[0]
            csv_orders_us["_prod_key"] = csv_orders_us[prod_col].astype(str).str.strip().str.upper()
            unique_prod_ids = sorted(set(csv_orders_us["_prod_key"].dropna().tolist()))

            # 4) Fetch origins from bi.erp_items
            items_origin = fetch_items_origin(unique_prod_ids)
            items_origin["_prod_key"] = items_origin["product_id"].astype(str).str.strip().str.upper()
            items_origin = items_origin.drop_duplicates(subset=["_prod_key"])

            # Stash for reuse in the compute step
            st.session_state.ddp_items_origin = items_origin
            st.session_state.ddp_csv_orders_us = csv_orders_us
            st.session_state.ddp_prod_col = prod_col

        st.success("✅ Fetched DDP config and product origins.")

    except Exception as e:
        st.error(f"❌ Error during DDP fetch: {e}")

# --- If we have config & origins, compute the DDP recalculation (US only) ---
if (
    "ddp_config" in st.session_state and isinstance(st.session_state.ddp_config, pd.DataFrame) and
    "ddp_items_origin" in st.session_state and isinstance(st.session_state.ddp_items_origin, pd.DataFrame) and
    "ddp_csv_orders_us" in st.session_state
):

    ddp_cfg = st.session_state.ddp_config.copy()
    items_origin = st.session_state.ddp_items_origin.copy()
    csv_orders = st.session_state.ddp_csv_orders_us.copy()
    prod_col = st.session_state.ddp_prod_col

    st.markdown("#### Config preview")
    st.dataframe(ddp_cfg.head(50), width='stretch', height=220)

    st.markdown("#### Product Origin Preview")
    st.dataframe(items_origin.head(50), width='stretch', height=220)

    # --- Build the line-level base frame (Orders only, US only) ---
    csv_orders["_prod_key"] = csv_orders[prod_col].astype(str).str.strip().str.upper()
    csv_orders["_ship_country"] = csv_orders["Shipping Country"].astype(str).str.strip().str.upper()

    # Bring origin country (made_in)
    enrich = csv_orders.merge(
        items_origin[["_prod_key", "made_in"]],
        on="_prod_key", how="left"
    )
    enrich["_origin_country"] = enrich["made_in"].astype(str).str.strip().str.upper()

    # Prepare config join keys
    ddp_cfg["_brand_code"]     = ddp_cfg["brand_code"].astype(str).str.strip().str.upper()
    ddp_cfg["_ship_country"]   = ddp_cfg["shipping_country"].astype(str).str.strip().str.upper()
    ddp_cfg["_origin_country"] = ddp_cfg["origin_country"].astype(str).str.strip().str.upper()
    ddp_cfg["_src_co"]         = ddp_cfg["source_company"].astype(str).str.strip().str.upper()

    # Filter config to our context (brand_code + source_company), ignore oms_location_name
    _brand_code = (st.session_state.get("ddp_brand_code") or brand).strip().upper()
    _src_company = erpEntity.strip().upper()
    ctx_cfg = ddp_cfg[(ddp_cfg["_brand_code"] == _brand_code) & (ddp_cfg["_src_co"] == _src_company)].copy()

    # Keep the latest application_date per (ship, origin)
    if not ctx_cfg.empty and "application_date" in ctx_cfg.columns:
        ctx_cfg["application_date"] = pd.to_datetime(ctx_cfg["application_date"], errors="coerce")
        ctx_cfg = (
            ctx_cfg
            .sort_values(["_ship_country", "_origin_country", "application_date"],
                         ascending=[True, True, False])
            .drop_duplicates(subset=["_ship_country", "_origin_country"], keep="first")
        )

    # Join on (shipping_country, origin_country)
    staged = enrich.merge(
        ctx_cfg,
        left_on=["_ship_country", "_origin_country"],
        right_on=["_ship_country", "_origin_country"],
        how="left",
        suffixes=("", "_cfg")
    )

    # --- Compute DDP using Sell Price as base (no currency logic) ---
    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    staged["Sell Price"]       = _to_num(staged.get("Sell Price"))
    staged["DDP Services"]     = _to_num(staged.get("DDP Services"))
    staged["ddp_fix"]          = _to_num(staged.get("ddp_fix")).fillna(0.0)
    staged["ddp_perc"]         = _to_num(staged.get("ddp_perc")).fillna(0.0)
    staged["treshold_amount"]  = _to_num(staged.get("treshold_amount")).fillna(0.0)

    # Base = amount_including_vat = Sell Price (original currency)
    base = staged["Sell Price"]
    thr  = staged["treshold_amount"]
    perc = staged["ddp_perc"]
    fix  = staged["ddp_fix"]

    # ddp_recalc = ddp_fix + ddp_perc% * base    if base >= threshold
    #             = ddp_fix                       otherwise
    cond = base.notna() & (base >= thr)
    staged["ddp_recalc"] = np.where(cond, fix + (perc * base), fix)

    # Compare against CSV DDP Services (both in original currency)
    staged["ddp_delta"] = (staged["ddp_recalc"] - staged["DDP Services"].fillna(0)).round(2)

    # Display
    st.markdown("#### Recalculated DDP Preview")
    show_cols = [
        "Order Number", prod_col, "_ship_country", "_origin_country",
        "Sell Price", "ddp_fix", "ddp_perc", "treshold_amount",
        "ddp_recalc", "DDP Services", "ddp_delta", "application_date"
    ]
    show_cols = [c for c in show_cols if c in staged.columns]
    st.dataframe(staged.head(100)[show_cols], width='stretch', height=380)


    # Mismatches table (non-zero delta)
    mismatch_tbl = staged.loc[staged["ddp_delta"].abs() > 0.01, show_cols].copy()
    st.markdown("#### Lines with DDP mismatch (|Δ| > 0.01)")
    st.caption(f"{len(mismatch_tbl):,} line(s)")
    st.dataframe(mismatch_tbl.head(2000), width='stretch', height=360)

    # Persist for downstream use / export
    st.session_state.ddp_recalc_frame = staged

else:
    st.info("👆 Click **Fetch DDP Config & Origins** to compute the per-line DDP.")

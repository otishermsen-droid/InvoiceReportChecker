# common.py
import os
import io
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import pandas as pd
from google.cloud import bigquery

from typing import List, Optional

import streamlit as st
from google.oauth2 import service_account

# =====================
# Setup & configuration
# =====================

LOG_DIR = Path("logs/invoicingReport")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ],
)

# Env/config (set your project/dataset/table via env or sidebar)
BQ_PROJECT = os.environ.get("BQ_PROJECT", "tlg-business-intelligence-prd")
BQ_DATASET = os.environ.get("BQ_DATASET", "bi")
BQ_TABLE = os.environ.get("BQ_TABLE", "orders_returns_new")

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)


"""
Brand-specific CSV header arrays
- Some brands provide different column orders or additional columns
- We select the expected headers list per brand, preserving the app's canonical names
"""

# Default/canonical headers (without the trailing extra-collector)
_DEFAULT_HEADERS_BASE: List[str] = [
    "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
    "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
    "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
    "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
    "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
    "Qualità Reso.","COGS x Qty"
]

# Per-brand header arrays. Provide only differences from default; if not present, default is used.
# IMPORTANT: Do not include the trailing "__extra__" here; it will be appended automatically.
BRAND_HEADERS: Dict[str, List[str]] = {
    # Example placeholders; replace when full specs are provided
    "FO": _DEFAULT_HEADERS_BASE,
    "AL": _DEFAULT_HEADERS_BASE,
    "HB": _DEFAULT_HEADERS_BASE,
    "RC": _DEFAULT_HEADERS_BASE,
    "FU": _DEFAULT_HEADERS_BASE,
    "MO": _DEFAULT_HEADERS_BASE,
    "MA": _DEFAULT_HEADERS_BASE,
    "BO": _DEFAULT_HEADERS_BASE,
    "FA": _DEFAULT_HEADERS_BASE,
    "CA":  [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.", "Channel", "COGS x Qty"
    ],
    "AT": [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso."
    ],
    "PJ": [
        "StoreID", "Export Date", "Transaction ID", "Row ID", "Date",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.", "Matricolari"
    ],
    "FE": [
        "Shipping Country","EAN","Packing List","Packing list date",
        "Model","Fabric","Color","Size Type","Size","Yoox Code","Category",
        "Brand","Season","Qty","Transaction ID","Date","Currency",
        "Original Price Value","Original Price","Discount %","Discount Value","VAT%",
        "Sales Tax","DDP Services","GMV Net VAT","Value for invoice","Type","Item Status",
        "% TLG FEE","COGS","Export Date","Row ID",
        "Exchange rate","Sales Order Date","Order Number"
    ],
    "CV": [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.", "COGS x Qty", "OMS Location Name", "Endless Aisle"
    ],
    "PL": [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.","COGS x Qty","P.IVA The Level","achillepinto_model","_extra4_","_extra5_"
    ],
    "HE": [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.","COGS x Qty","Warehouse Location", "Year Month Type"
    ]
}


def get_supported_brands() -> List[str]:
    """Return supported brand codes for the dropdown."""
    brands = sorted({code.strip().upper() for code in BRAND_HEADERS.keys()})
    return brands or ["PJ"]


def get_headers_for_brand(brand: Optional[str], erp_entity: Optional[str] = None) -> List[str]:
    """Return the expected CSV headers array for a given brand.

    Always appends a trailing "__extra__" to collect overflow fields when present.
    """
    key = (brand or "").strip().upper()
    ent = (erp_entity or "").strip().upper()
    base = BRAND_HEADERS.get(key, _DEFAULT_HEADERS_BASE)
    # ERP-specific adjustments
    if key == "CV" and ent == "TLG_USA":
        # For TLG_USA, CV has one fewer column: drop 'COGS x Qty'
        base = [c for c in base if c != "COGS x Qty"]
    # Ensure copy and append extra collector
    headers = list(base)
    if "__extra__" not in headers:
        headers.append("__extra__")
    return headers


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_invoicing_report(file: Union[io.BytesIO, str], brand: Optional[str] = None, erp_entity: Optional[str] = None) -> pd.DataFrame:
    import csv
    headers = get_headers_for_brand(brand, erp_entity=erp_entity)
    file.seek(0)
    df = pd.read_csv(
        file,
        sep=";",
        decimal=",",
        header=None,
        names=headers,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        escapechar=None,
        encoding="utf-8"
    )
    if len(df.columns) != len(headers):
        logging.warning(
            "CSV column count (%s) does not match expected headers (%s).",
            len(df.columns), len(headers)
        )
    if "__extra__" in df.columns and df["__extra__"].isna().all():
        df = df.drop(columns=["__extra__"])

    # Parse dates (EU format) with explicit formats to avoid warnings
    def _parse_eu_date_series(s: pd.Series) -> pd.Series:
        import warnings
        s = s.astype(str).str.strip()
        # Normalize common artifacts (non-breaking spaces, trailing .0 from float-casts)
        s = s.str.replace("\u00A0", "", regex=False)
        s = s.str.replace(r"\.0+$", "", regex=True)
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y",
            "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d",
            "%Y%m%d", "%d%m%Y",
            "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"
        ]
        parsed = None
        for fmt in formats:
            try:
                parsed = pd.to_datetime(s, format=fmt, errors="coerce")
                # Accept if any values parsed
                if parsed.notna().any():
                    return parsed
            except Exception:
                pass
        # Fallback with warning suppressed
        # Final fallback: explicit two-digit year, dayfirst
        try:
            fallback = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
            if fallback.notna().any():
                return fallback
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return pd.to_datetime(s, errors="coerce", dayfirst=True)

    for dcol in ["Export Date", "Date", "Data Fattura"]:
        if dcol in df.columns:
            df[dcol] = _parse_eu_date_series(df[dcol])

    # Do not backfill Date from Export Date

    # Brand-specific numeric normalization (before coercion)
    brand_key = (brand or "").strip().upper()
    if brand_key == "AT" or brand_key == "FO" or brand_key == "AL" or brand_key == "CV":
        # AT brand: values like "374,180" (comma decimal). Normalize to dot decimals.
        if "COGS" in df.columns:
            df["COGS"] = (
                df["COGS"].astype(str)
                .str.replace("\u00A0", "", regex=False)  # non-breaking spaces
                .str.replace(" ", "", regex=False)       # regular spaces
                .str.replace(".", "", regex=False)       # thousands separator
                .str.replace(",", ".", regex=False)      # decimal comma -> dot
            )

    num_cols = [
        "Qty", "COGS", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2"
    ]
    df = _coerce_numeric(df, num_cols)
    return df

# to fix TLG FEE
def sanity_check_tlg_fee(
    df: pd.DataFrame,
    atol: float = 0.01,
    rtol: float = 0.01,
    override_percent: Optional[float] = None,
) -> pd.DataFrame:
    req = ["TLG Fee", "GMV Net VAT", "% TLG FEE"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    # Decide which percent to use: optional override from UI; otherwise prefer recalc column, then original column
    if override_percent is not None and pd.notna(override_percent) and float(override_percent) > 0:
        percent_series = pd.Series(float(override_percent), index=df.index)
    else:
        source_col = "recalc_%TLG FEE" if "recalc_%TLG FEE" in df.columns else "% TLG FEE"
        # Avoid division by zero and NaN by replacing explicit 0 with NA to enable abs_ok tolerance
        percent_series = df[source_col].replace(0, pd.NA)
    df["expected_tlg_fee"] = (df["GMV Net VAT"] * (percent_series / 100)).round(2)
    diff = (df["TLG Fee"] - df["expected_tlg_fee"]).abs()
    rel_ok = diff <= (df["expected_tlg_fee"].abs() * rtol).fillna(0)
    abs_ok = diff <= atol
    df["tlg_fee_match"] = rel_ok | abs_ok
    mismatches = df.loc[~df["tlg_fee_match"]].copy()
    mismatches["delta"] = mismatches["TLG Fee"] - \
        mismatches["expected_tlg_fee"]
    return mismatches

def sanity_check_cogs(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    # COGS ≈ GMV Net VAT - TLG Fee
    req = ["COGS", "GMV Net VAT", "TLG Fee"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    _coerce_numeric(df, req) 

    # Compute expectations
    df["expected_cogs"] = df["GMV Net VAT"] - df["TLG Fee"]

    # Safe numeric diff
    diff = (df["COGS"] - df["expected_cogs"]).abs()

    # Build tolerance checks (these produce nullable booleans if NaNs exist)
    rel_ok = diff <= (df["expected_cogs"].abs() * rtol)
    abs_ok = diff <= atol

    # Collapse NA to False so the mask is pure bool
    df["cogs_match"] = (rel_ok.fillna(False) | abs_ok.fillna(False))

    # Now select mismatches with a solid bool mask
    mismatches = df.loc[~df["cogs_match"]].copy()

    # Compute delta inside the subset (avoids weird cross-alignment surprises)
    mismatches["delta"] = mismatches["COGS"] - mismatches["expected_cogs"]

    return mismatches


def sanity_checks_ddp_tax(df: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    req = ["% Tax", "DDP Services"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    mismatches = df[(df["% Tax"] > tol) & (df["DDP Services"] > tol)].copy()
    return mismatches


def sanity_check_gmv_eur(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    req = ["Sell Price", "Discount", "DDP Services", "GMV EUR", "Exchange rate"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    # Only check rows where all required columns are present (not null)
    valid_mask = df[req].notnull().all(axis=1)
    df_valid = df[valid_mask].copy()
    df_valid["expected_gmv"] = (df_valid["Sell Price"] - df_valid["Discount"] - df_valid["DDP Services"]) / df_valid["Exchange rate"]
    diff = (df_valid["GMV EUR"] - df_valid["expected_gmv"]).abs()
    rel_ok = diff <= (df_valid["expected_gmv"].abs() * rtol).fillna(0)
    abs_ok = diff <= 0.01  # Always accept a difference of 0.01
    df_valid["gmv_match"] = rel_ok | abs_ok
    mismatches = df_valid.loc[~df_valid["gmv_match"]].copy()
    mismatches["delta"] = mismatches["GMV EUR"] - mismatches["expected_gmv"]
    return mismatches

def sanity_check_gmv_net(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    req = ["% Tax", "GMV EUR", "GMV Net VAT"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    # Handle 0% tax correctly (should not set to NA)
    percent = df["% Tax"].fillna(0)
    df["expected_gmv_net"] = df["GMV EUR"] / (1 + percent / 100)
    diff = (df["GMV Net VAT"] - df["expected_gmv_net"]).abs()
    rel_ok = diff <= (df["expected_gmv_net"].abs() * rtol)
    abs_ok = diff <= atol
    df["gmv_net_match"] = (rel_ok.fillna(False) | abs_ok.fillna(False))
    mismatches = df.loc[~df["gmv_net_match"]].copy()
    mismatches["delta"] = mismatches["GMV Net VAT"] - mismatches["expected_gmv_net"]
    return mismatches

def fetch_orders_returns(start_date: str, end_date: str, brand: str, source_company: str) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE brand = @brand
        AND source_company = @source_company
        AND (
            (row_type = 'O'  AND DATE(invoice_creation_date)     BETWEEN @start_date AND @end_date)
            OR
            (row_type = 'R' AND DATE(credit_note_creation_date) BETWEEN @start_date AND @end_date)
        )
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            bigquery.ScalarQueryParameter("source_company", "STRING", source_company),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def fetch_brand_code(brand_input: str) -> Optional[str]:
    """
    Resolve a 2-letter brand input (e.g., 'PJ') to the canonical brand_code
    from `config.brand`. Tries a few common column names defensively.
    Returns None if not found.
    """
    # Being liberal about column names: brand, brand_code, acronym, code
    sql = f"""
        SELECT brand_id AS brand_code
        FROM `{BQ_PROJECT}.config.brand`
        WHERE brand_code = @brand_input
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand_input", "STRING", brand_input.strip().upper())
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    if df.empty or df["brand_code"].isna().all():
        return None
    return str(df.iloc[0]["brand_code"]).strip()


def fetch_ddp_config(brand_code: str, source_company: str) -> pd.DataFrame:
    """
    Pull DDP config rows for the given brand_code + ERP entity (source_company)
    from `config.erp_ico_ddp`.
    """
    sql = f"""
        SELECT
          brand_code,
          shipping_country,
          oms_location_name,
          duty_recalculation,
          ddp_fix,
          ddp_perc,
          treshold_amount,
          currency_code,
          application_date,
          origin_country,
          source_company
        FROM `{BQ_PROJECT}.config.erp_ico_ddp`
        WHERE UPPER(TRIM(brand_code)) = UPPER(@brand_code)
          AND UPPER(TRIM(source_company)) = UPPER(@source_company)
          AND shipping_country = 'US'
    """ 
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand_code", "STRING", brand_code),
            bigquery.ScalarQueryParameter("source_company", "STRING", source_company),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def fetch_items_origin(product_ids: List[str]) -> pd.DataFrame:
    """
    Fetch `product_id` and `made_in` (origin) from `bi.erp_items` for the provided IDs.
    """
    if not product_ids:
        return pd.DataFrame(columns=["product_id", "made_in"])
    sql = f"""
        SELECT DISTINCT
          CAST(product_id AS STRING) AS product_id,
          CAST(made_in AS STRING)    AS made_in
        FROM `{BQ_PROJECT}.bi.erp_items`
        WHERE CAST(product_id AS STRING) IN UNNEST(@ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("ids", "STRING", [str(x) for x in product_ids])
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


# ==============================
# TLG Fee config & YTD KPI fetch
# ==============================
def fetch_tlg_fee_config(brand: str) -> pd.DataFrame:
    """Fetch TLG brand fee configuration rows for a given brand.

    Expected columns in table:
    brand, location_code, shipping_country_exclusion, shipping_country_inclusion,
    erp_entity, fee_perc, fee_calc_base, threshold_base, min_threshold_ytd,
    max_threshold_ytd, endless_aisle, currency
    """
    sql = f"""
        SELECT
          brand,
          location_code,
          shipping_country_exclusion,
          shipping_country_inclusion,
          erp_entity,
          CAST(fee_perc AS FLOAT64) AS fee_perc,
          fee_calc_base,
          threshold_base,
          CAST(min_threshold_ytd AS FLOAT64) AS min_threshold_ytd,
          CAST(max_threshold_ytd AS FLOAT64) AS max_threshold_ytd,
          endless_aisle,
          currency
        FROM `{BQ_PROJECT}.config.brand_fee`
        WHERE UPPER(TRIM(brand)) = UPPER(@brand)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand.strip().upper()),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def fetch_ytd_totals_for_brand(brand: str) -> pd.DataFrame:
    """Fetch YTD GMV/NMV totals for the provided brand for the current Rome year."""
    sql = f"""
        SELECT
          EXTRACT(YEAR FROM DATE(CASE
            WHEN row_type = 'O' THEN invoice_creation_date
            WHEN row_type = 'R' THEN credit_note_creation_date
          END)) AS year,
          ROUND(SUM(CASE WHEN row_type = 'O' THEN gmv_eur ELSE 0 END), 2) AS total_gmv_eur,
          ROUND(SUM(CASE WHEN row_type = 'R' THEN rmv_eur ELSE 0 END), 2) AS total_returns_eur,
          ROUND(
            SUM(CASE WHEN row_type = 'O' THEN gmv_eur ELSE 0 END)
            + SUM(CASE WHEN row_type = 'R' THEN rmv_eur ELSE 0 END),
          2) AS total_nmv_eur
        FROM `{BQ_PROJECT}.bi.orders_returns_new`
        WHERE brand = @brand
          AND EXTRACT(YEAR FROM DATE(CASE
                WHEN row_type = 'O' THEN invoice_creation_date
                WHEN row_type = 'R' THEN credit_note_creation_date
              END)) = EXTRACT(YEAR FROM CURRENT_DATE('Europe/Rome'))
        GROUP BY year
        ORDER BY year
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand.strip().upper()),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def decide_fee_percent_from_config(
    fee_config_df: pd.DataFrame,
    ytd_df: pd.DataFrame | None
) -> float | None:
    """Decide which fee percentage to apply based on optional YTD thresholds.

    Logic:
    - If threshold columns contain numeric values and fee_calc_base/threshold_base specify GMV or NMV,
      choose the row where YTD total is within [min_threshold_ytd, max_threshold_ytd].
    - If multiple rows match, take the first by min_threshold_ytd ascending.
    - If no thresholds present, fall back to the first non-null fee_perc.
    Returns a float percent or None if not determinable.
    """
    if fee_config_df is None or fee_config_df.empty:
        return None

    # Normalize types
    working = fee_config_df.copy()
    for c in ["fee_perc", "min_threshold_ytd", "max_threshold_ytd"]:
        if c in working.columns:
            working[c] = pd.to_numeric(working[c], errors="coerce")
    calc_base = (working.get("fee_calc_base") or pd.Series([], dtype=str)).astype(str).str.upper()
    thr_base = (working.get("threshold_base") or pd.Series([], dtype=str)).astype(str).str.upper()
    working["fee_calc_base_norm"] = calc_base
    working["threshold_base_norm"] = thr_base

    # If we have YTD totals and at least one row with thresholds, attempt threshold selection
    selected_fee: float | None = None
    if ytd_df is not None and not ytd_df.empty:
        ytd_row = ytd_df.iloc[-1]
        total_gmv = pd.to_numeric(ytd_row.get("total_gmv_eur"), errors="coerce")
        total_nmv = pd.to_numeric(ytd_row.get("total_nmv_eur"), errors="coerce")
        # Try NMV then GMV according to row base
        candidates = working.dropna(subset=["min_threshold_ytd", "max_threshold_ytd", "fee_perc"]).copy()
        if not candidates.empty:
            # Evaluate threshold value based on threshold_base
            def within_threshold(row) -> bool:
                base = str(row.get("threshold_base_norm", "")).upper()
                value = None
                if base == "NMV" and pd.notna(total_nmv):
                    value = float(total_nmv)
                elif base == "GMV" and pd.notna(total_gmv):
                    value = float(total_gmv)
                else:
                    return False
                return (value >= float(row["min_threshold_ytd"])) and (value <= float(row["max_threshold_ytd"]))

            matches = candidates[candidates.apply(within_threshold, axis=1)].copy()
            if not matches.empty:
                matches = matches.sort_values(by=["min_threshold_ytd"])  # deterministic
                selected_fee = float(matches.iloc[0]["fee_perc"]) if pd.notna(matches.iloc[0]["fee_perc"]) else None

    # Fallback: first fee_perc available
    if selected_fee is None:
        first = working.dropna(subset=["fee_perc"]).head(1)
        if not first.empty:
            val = first.iloc[0]["fee_perc"]
            selected_fee = float(val) if pd.notna(val) else None

    return selected_fee


def _normalize_yes_no_flag(value: Any) -> int | None:
    """Normalize assorted truthy/falsey values to 1/0, or None if unknown."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip().upper()
    if s in {"1", "Y", "YES", "TRUE"}:
        return 1
    if s in {"0", "N", "NO", "FALSE"}:
        return 0
    # Fallback for numeric-like
    try:
        n = int(float(s))
        if n in (0, 1):
            return n
    except Exception:
        pass
    return None


def _parse_list_field(field: Any) -> list[str]:
    """Parse a config list field that may contain comma-separated values, '*' wildcard, or be null.

    Returns list of uppercased trimmed tokens; '*' returns ['*'].
    """
    if field is None or (isinstance(field, float) and pd.isna(field)):
        return []
    s = str(field).strip()
    if not s:
        return []
    if s == "*":
        return ["*"]
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    return parts


from typing import Any, Optional
import pandas as pd

def apply_tlg_fee_config_per_row(
    df: pd.DataFrame,
    brand: str,
    fee_config_df: pd.DataFrame,
    erp_entity: Optional[str],
    static_ytd_gmv: float | int | None,
    static_ytd_nmv: float | int | None,
) -> pd.DataFrame:
    """Compute per-row TLG fee percent using static YTD totals and config filters.

    Rules:
    - Use one pair of YTD totals (GMV/NMV). No running totals.
    - Filters (with '*' or blank = wildcard):
        * brand: match exact unless '*' / blank in config
        * erp_entity: match exact unless '*' / blank
        * location_code: match `OMS Location Name` unless '*' / blank
        * shipping_country inclusion/exclusion: honor lists unless '*'
        * endless_aisle (brand == 'CV'): match 0/1 if set; ignore if null/blank
    - Thresholds: if present, pick row whose [min,max] contains the selected base (threshold_base or fee_calc_base).
      If none match, fall back to no-threshold rows with a defined fee.
    - Write only `recalc_%TLG FEE`. Do not change existing fee columns.
    """
    if df is None or df.empty or fee_config_df is None or fee_config_df.empty:
        return df

    working = fee_config_df.copy()

    # Normalize string-ish columns
    for col in [
        "brand", "location_code", "shipping_country_exclusion",
        "shipping_country_inclusion", "erp_entity", "fee_calc_base",
        "threshold_base", "endless_aisle"
    ]:
        if col in working.columns:
            working[col] = working[col].astype(str).str.strip()

    # Numerics
    for col in ["fee_perc", "min_threshold_ytd", "max_threshold_ytd"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    # Treat null-like as NA so they behave as wildcards
    def _nullify(series: pd.Series) -> pd.Series:
        return series.replace(
            {"": pd.NA, "null": pd.NA, "None": pd.NA, "none": pd.NA, "NaN": pd.NA, "nan": pd.NA}
        )

    for col in ["brand", "location_code", "shipping_country_exclusion", "shipping_country_inclusion", "erp_entity", "endless_aisle"]:
        if col in working.columns:
            working[col] = _nullify(working[col])


    # Pre-parse inclusion/exclusion lists
    if "shipping_country_inclusion" in working.columns:
        working["_incl_list"] = working["shipping_country_inclusion"].apply(_parse_list_field)
    else:
        working["_incl_list"] = [[] for _ in range(len(working))]

    if "shipping_country_exclusion" in working.columns:
        working["_excl_list"] = working["shipping_country_exclusion"].apply(_parse_list_field)
    else:
        working["_excl_list"] = [[] for _ in range(len(working))]

    print(working)

    # Static YTD bases
    ytd_gmv = float(static_ytd_gmv or 0.0)
    ytd_nmv = float(static_ytd_nmv or 0.0)

    print(ytd_gmv)
    print(ytd_nmv)

    # Helpers
    def _erp_ok(x: Any, ent: Optional[str]) -> bool:
        if ent is None or (isinstance(ent, float) and pd.isna(ent)) or str(ent).strip() == "":
            return True
        if pd.isna(x):
            return True
        s = str(x).strip()
        return s == "*" or s.upper() == str(ent).strip().upper()

    def _brand_ok(x: Any, b: str) -> bool:
        if pd.isna(x):
            return True
        s = str(x).strip()
        return s == "*" or s.upper() == str(b).strip().upper()

    def _loc_ok(x: Any, omsv: Optional[str]) -> bool:
        if pd.isna(x):
            return True
        s = str(x).strip()
        if s == "*":
            return True
        if omsv is None or (isinstance(omsv, float) and pd.isna(omsv)) or str(omsv).strip() == "":
            return False
        return s.upper() == str(omsv).strip().upper()

    def _ship_match(incl_list: list[str], excl_list: list[str], ship_val: str) -> bool:
        # Guard to avoid elementwise warnings; coerce to Python list/set
        incl = list(incl_list) if isinstance(incl_list, (list, tuple)) else []
        excl = list(excl_list) if isinstance(excl_list, (list, tuple)) else []
        ship = (ship_val or "").strip().upper()

        if incl and incl != ["*"]:
            # set lookup for speed
            if ship not in set(incl):
                return False
        if excl and excl != ["*"]:
            if ship in set(excl):
                return False
        return True

    def _threshold_ok(cfg_row: pd.Series) -> bool:
        min_thr = cfg_row.get("min_threshold_ytd")
        max_thr = cfg_row.get("max_threshold_ytd")
        base = str(cfg_row.get("threshold_base") or cfg_row.get("fee_calc_base") or "").strip().upper()
        if pd.isna(min_thr) and pd.isna(max_thr):
            return False
        value = ytd_gmv if base == "GMV" else ytd_nmv if base == "NMV" else ytd_gmv
        try:
            low = float(min_thr) if not pd.isna(min_thr) else float("-inf")
            high = float(max_thr) if not pd.isna(max_thr) else float("inf")
            return float(value) >= low and float(value) <= high
        except Exception:
            return False

    out = df.copy(deep=False)
    # Normalized per-row fields
    out["_ship"] = out.get("Shipping Country", pd.Series(index=out.index)).astype(str).str.strip().str.upper()
    if "OMS Location Name" in out.columns:
        out["_oms"] = out["OMS Location Name"].astype(str).str.strip().str.upper()
    else:
        out["_oms"] = None

    cv_brand = (str(brand).strip().upper() == "CV")
    if cv_brand and "Endless Aisle" in out.columns:
        out["_ea"] = out["Endless Aisle"].apply(_normalize_yes_no_flag)
    else:
        out["_ea"] = None

    print(out)

    perc_values: list[float | None] = []

    for _, row in out.iterrows():
        ship = row.get("_ship") or ""
        omsv = row.get("_oms")
        endless_val = row.get("_ea")

        cand = working

        # Brand filter (with wildcard/blank support)
        if "brand" in cand.columns:
            cand = cand[cand["brand"].apply(lambda x: _brand_ok(x, brand))]

        # ERP entity
        if "erp_entity" in cand.columns:
            cand = cand[cand["erp_entity"].apply(lambda x: _erp_ok(x, erp_entity))]

        # Location
        if "location_code" in cand.columns:
            cand = cand[cand["location_code"].apply(lambda x: _loc_ok(x, omsv))]

        # Shipping include/exclude
        if "_incl_list" in cand.columns and "_excl_list" in cand.columns:
            cand = cand[cand.apply(lambda r: _ship_match(r["_incl_list"], r["_excl_list"], ship), axis=1)]

        # Endless Aisle (only for CV brand)
        if cv_brand and "endless_aisle" in cand.columns:
            def endless_ok(x: Any) -> bool:
                if pd.isna(x) or str(x).strip() == "":
                    return True
                cfg_flag = _normalize_yes_no_flag(x)
                if cfg_flag is None:
                    return True
                if endless_val is None:
                    return False
                return int(cfg_flag) == int(endless_val)
            cand = cand[cand["endless_aisle"].apply(endless_ok)]

        fee = None
        if not cand.empty:
            # Prefer matching threshold rows
            with_thr = cand[cand.apply(_threshold_ok, axis=1)].copy()
            if not with_thr.empty:
                with_thr["_low"] = with_thr["min_threshold_ytd"].fillna(0.0)
                # deterministic tie-break: lowest min_threshold then highest fee
                with_thr = with_thr.sort_values(by=["_low", "fee_perc"], ascending=[True, False])
                fee = with_thr.iloc[0].get("fee_perc")
            else:
                # Fallback: rows without thresholds (or any with a fee)
                no_thr = cand[(cand["min_threshold_ytd"].isna()) & (cand["max_threshold_ytd"].isna())]
                pool = no_thr if not no_thr.empty else cand
                pool = pool.dropna(subset=["fee_perc"]).copy()
                if not pool.empty:
                    pool = pool.sort_values(by=["fee_perc"], ascending=[False])
                    fee = pool.iloc[0].get("fee_perc")

        perc_values.append(float(fee) if fee is not None and not pd.isna(fee) else None)

    out["recalc_%TLG FEE"] = pd.Series(perc_values, index=out.index)

    # cleanup temp cols
    out = out.drop(columns=[c for c in ["_ship", "_oms", "_ea"] if c in out.columns])
    return out



def fetch_ytd_totals_until_date(brand: str, end_date: str) -> pd.DataFrame:
    """Fetch YTD GMV/NMV totals for the given brand up to and including the provided end_date.

    end_date: string in YYYY-MM-DD format. Uses the calendar year of end_date.
    """
    sql = f"""
        SELECT
          EXTRACT(YEAR FROM DATE(CASE
            WHEN row_type = 'O' THEN invoice_creation_date
            WHEN row_type = 'R' THEN credit_note_creation_date
          END)) AS year,
          ROUND(SUM(CASE WHEN row_type = 'O' THEN gmv_eur ELSE 0 END), 2) AS total_gmv_eur,
          ROUND(SUM(CASE WHEN row_type = 'R' THEN rmv_eur ELSE 0 END), 2) AS total_returns_eur,
          ROUND(
            SUM(CASE WHEN row_type = 'O' THEN gmv_eur ELSE 0 END)
            + SUM(CASE WHEN row_type = 'R' THEN rmv_eur ELSE 0 END),
          2) AS total_nmv_eur
        FROM `{BQ_PROJECT}.bi.orders_returns_new`
        WHERE brand = @brand
          AND EXTRACT(YEAR FROM DATE(CASE
                WHEN row_type = 'O' THEN invoice_creation_date
                WHEN row_type = 'R' THEN credit_note_creation_date
              END)) = EXTRACT(YEAR FROM DATE(@end_date))
          AND DATE(CASE
                WHEN row_type = 'O' THEN invoice_creation_date
                WHEN row_type = 'R' THEN credit_note_creation_date
              END) <= DATE(@end_date)
        GROUP BY year
        ORDER BY year
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand.strip().upper()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()
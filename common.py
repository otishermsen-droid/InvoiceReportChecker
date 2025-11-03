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

from collections.abc import Iterable
import re
from typing import Optional, Any

from typing import Any, Optional
import pandas as pd

from typing import Optional, Any, Iterable
import pandas as pd
import numpy as np


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
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
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

"""
Brand rule declaring whether, on DDP/Tax coexistence, we keep TAX (zero DDP) or keep DDP (zero TAX).
- Keyed by 2-letter brand code
"""
BRAND_TAX_DDP_RULE: Dict[str, str] = {
    "CV": "DDP",
    "FE": "TAX",
    "RC": "TAX",
    "CA": "TAX",
    "AL": "DDP",
    "AT": "DDP",
    "HB": "TAX",
    "FO": "DDP",
    "HE": "DDP",
    "PJ": "DDP",
    "MA": "DDP",
    "PL": "DDP",
    "BO": "DDP",
    "MO": "DDP",
    "FU": "DDP",
}

def get_brand_rule(brand: Optional[str]) -> Optional[str]:
    key = (brand or "").strip().upper()
    rule = BRAND_TAX_DDP_RULE.get(key)
    rule_norm = str(rule).strip().upper() if rule is not None else None
    if rule_norm in {"TAX", "DDP"}:
        return rule_norm
    return None

def zero_conflicting_fields(
    df: pd.DataFrame,
    rule: Optional[str],
    scope: str = "coexist",
) -> tuple[pd.DataFrame, int]:
    """Zero the non-primary field per brand rule.

    - rule == 'TAX': keep Tax, zero DDP Services
    - rule == 'DDP': keep DDP, zero % Tax / VAT%
    - scope == 'coexist': only rows where both DDP>0 and Tax>0
    Returns: (updated_df, rows_affected)
    """
    if df is None or df.empty or rule not in {"TAX", "DDP"}:
        return df, 0

    out = df.copy()

    # Determine columns
    ddp_col = "DDP Services" if "DDP Services" in out.columns else None
    tax_primary = "% Tax" if "% Tax" in out.columns else None
    tax_alt = "VAT%" if "VAT%" in out.columns else None

    if ddp_col is None and (tax_primary is None and tax_alt is None):
        return out, 0

    # Build numeric series for masking
    ddp_num = pd.to_numeric(out[ddp_col], errors="coerce") if ddp_col else pd.Series([pd.NA] * len(out), index=out.index)
    tax_num = None
    if tax_primary is not None:
        tax_num = pd.to_numeric(out[tax_primary], errors="coerce")
    elif tax_alt is not None:
        tax_num = pd.to_numeric(out[tax_alt], errors="coerce")
    else:
        tax_num = pd.Series([pd.NA] * len(out), index=out.index)

    # Coexistence rows mask
    coexist_mask = (
        (ddp_num.fillna(0) > 0)
        & (tax_num.fillna(0) > 0)
    )

    target_mask = coexist_mask if scope == "coexist" else pd.Series([True] * len(out), index=out.index)
    rows_affected = int(target_mask.sum())
    if rows_affected == 0:
        return out, 0

    if rule == "TAX":
        # Zero DDP Services
        if ddp_col is not None:
            out.loc[target_mask, ddp_col] = 0
    elif rule == "DDP":
        # Zero both % Tax and VAT% if present
        if tax_primary is not None:
            out.loc[target_mask, tax_primary] = 0
        if tax_alt is not None:
            out.loc[target_mask, tax_alt] = 0

    return out, rows_affected

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


def _normalize_numeric_text_series(series: pd.Series) -> pd.Series:
    """Normalize mixed-format numeric text to standard dotted-decimal strings.

    Handles:
    - Currency symbols (€, $, £)
    - Non-breaking/regular spaces and tabs
    - Parentheses for negatives (e.g., (1.234,56))
    - Mixed thousands/decimal separators (dot/comma)
    """
    def normalize_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""

        # Normalize minus variants and detect parentheses negatives
        text = text.replace("−", "-")
        is_negative = text.startswith("(") and text.endswith(")")
        if is_negative:
            text = text[1:-1]

        # Remove currency symbols and whitespace artifacts
        text = (
            text.replace("\u00A0", "")  # NBSP
                .replace(" ", "")
                .replace("\t", "")
                .replace("€", "")
                .replace("$", "")
                .replace("£", "")
        )

        # Determine decimal separator by rightmost occurrence of '.' or ','
        last_dot = text.rfind(".")
        last_comma = text.rfind(",")

        if last_dot == -1 and last_comma == -1:
            # No explicit decimal separator: keep only digits and an optional leading '-'
            cleaned = re.sub(r"[^0-9-]", "", text)
        else:
            # Choose rightmost as decimal separator
            decimal_sep = "." if last_dot > last_comma else ","
            parts = text.rsplit(decimal_sep, 1)
            left = re.sub(r"[^0-9-]", "", parts[0])
            right = re.sub(r"[^0-9]", "", parts[1]) if len(parts) > 1 else ""
            cleaned = f"{left}.{right}" if right != "" else left

        if is_negative and not cleaned.startswith("-") and cleaned != "":
            cleaned = f"-{cleaned}"
        return cleaned

    return series.astype(object).apply(normalize_value)


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
    if brand_key in {"AT", "FO", "AL", "CV", "RC"}:
        # Legacy normalization kept for specific brands (COGS only)
        if "COGS" in df.columns:
            df["COGS"] = (
                df["COGS"].astype(str)
                .str.replace("\u00A0", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

    # Ferrari (FE/FA): normalize all numeric columns to handle currency symbols and mixed separators
    if brand_key in {"FE", "FA"}:
        for c in [
            "Qty", "COGS", "% Tax", "VAT%",
            "Original Price", "Original Price Value", "Sales Tax", "Discount Value",
            "Sell Price", "Discount", "DDP Services",
            "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2"
        ]:
            if c in df.columns:
                df[c] = _normalize_numeric_text_series(df[c])

    num_cols = [
        "Qty", "COGS", "% Tax", "VAT%", "Original Price", "Original Price Value", "Sales Tax", "Discount Value", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2"
    ]
    df = _coerce_numeric(df, num_cols)

    # HE-specific: scale selected columns by dividing by 100 for internal use
    if brand_key == "HE":
        he_cols = [
            "Qty", "% Tax", "Original Price", "Sell Price", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2", "COGS x Qty"
        ]
        for c in he_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        # Exchange rate uses a different scaling factor
        if "Exchange rate" in df.columns:
            df["Exchange rate"] = pd.to_numeric(df["Exchange rate"], errors="coerce") / 100000.0
    
    # RC-specific enrichment: CODICE BRAND from bi.erp_items.MFVC via Product ID
    if brand_key == "RC":
        try:
            # Always ensure the column exists, even if lookup yields no rows
            if "CODICE BRAND" not in df.columns:
                df["CODICE BRAND"] = pd.NA
            if "Product ID" in df.columns:
                prod_ids = (
                    df["Product ID"].astype(str).str.strip()
                    .dropna().unique().tolist()
                )
                if prod_ids:
                    parts_df = fetch_items_rc_brand_parts(prod_ids)
                    if not parts_df.empty:
                        df["__prod_key__"] = df["Product ID"].astype(str).str.strip()
                        parts_df["__prod_key__"] = parts_df["product_id"].astype(str).str.strip()
                        # If multiple rows exist for a product, keep only the first
                        parts_df = parts_df.drop_duplicates(subset=["__prod_key__"], keep="first")
                        df = df.merge(
                            parts_df[["__prod_key__", "model", "variant", "fabric", "color"]],
                            on="__prod_key__",
                            how="left",
                        )
                        # Compose CODICE BRAND as {model}-{variant}-{fabric}{color}
                        def _compose_brand_code(row: pd.Series) -> Optional[str]:
                            m = str(row.get("model", "") or "").strip()
                            v = str(row.get("variant", "") or "").strip()
                            f = str(row.get("fabric", "") or "").strip()
                            c = str(row.get("color", "") or "").strip()
                            if not any([m, v, f, c]):
                                return None
                            left = f"{m}-{v}" if m or v else ""
                            right = f"{f}{c}"
                            if left and right:
                                return f"{left}-{right}"
                            return left or right or None
                        composed = df.apply(_compose_brand_code, axis=1)
                        df["CODICE BRAND"] = composed.where(pd.notna(composed) & (composed != ""), df["CODICE BRAND"])  
                        # Cleanup temp columns
                        drop_cols = [c for c in ["__prod_key__", "model", "variant", "fabric", "color"] if c in df.columns]
                        if drop_cols:
                            df = df.drop(columns=drop_cols)
        except Exception as e:
            logging.warning("RC enrichment for CODICE BRAND failed: %s", e)
    return df

# to fix TLG FEE
def sanity_check_tlg_fee(
    df: pd.DataFrame,
    atol: float = 0.01,
    rtol: float = 0.01,
    override_percent: Optional[float] = None,
) -> pd.DataFrame:
    # Check if TLG Fee column exists - if not, return empty DataFrame
    if "TLG Fee" not in df.columns:
        return pd.DataFrame()
    
    req = ["TLG Fee", "GMV Net VAT", "% TLG FEE"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    
    # Decide which percent to use: optional override from UI; otherwise prefer recalc column, then original column
    if override_percent is not None and pd.notna(override_percent) and float(override_percent) > 0:
        percent_series = pd.Series(float(override_percent), index=df.index)
        # Override is assumed to be in percentage format, so divide by 100
        percent_series = percent_series / 100
    else:
        source_col = "recalc_%TLG FEE" if "recalc_%TLG FEE" in df.columns else "% TLG FEE"
        # Use 0 instead of NA for zero/blank percents
        percent_series = df[source_col]

        # HE brand: values were divided by 100 on load, so % TLG FEE is already decimal
        brand_key = st.session_state.get("brand") if "brand" in st.session_state else None
        he_decimal_mode = isinstance(brand_key, str) and brand_key.strip().upper() == "HE"

        # Handle different formats
        if source_col == "recalc_%TLG FEE":
            # recalc_%TLG FEE is already in decimal format
            pass
        else:
            percent_series = percent_series if he_decimal_mode else (percent_series / 100)

    # Ensure missing percents or GMV yield expected fee of 0 (not NaN)
    df["expected_tlg_fee"] = (
        df["GMV Net VAT"].fillna(0) * percent_series.fillna(0)
    ).round(2)
    diff = (df["TLG Fee"] - df["expected_tlg_fee"]).abs()
    rel_ok = diff <= (df["expected_tlg_fee"].abs() * rtol).fillna(0)
    abs_ok = diff <= atol
    df["tlg_fee_match"] = rel_ok | abs_ok
    mismatches = df.loc[~df["tlg_fee_match"]].copy()
    mismatches["delta"] = mismatches["TLG Fee"] - \
        mismatches["expected_tlg_fee"]
    return mismatches

def sanity_check_cogs(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    # Skip COGS check for Ferrari (FE): TLG Fee is not available, so cannot compute expected COGS
    try:
        brand_key = st.session_state.get("brand")
        if isinstance(brand_key, str) and brand_key.strip().upper() == "FE":
            return pd.DataFrame()
    except Exception:
        # If session state is unavailable, proceed with generic logic
        pass

    # COGS ≈ GMV Net VAT - TLG Fee
    req = ["COGS", "GMV Net VAT"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check if we have TLG Fee column or need to calculate it
    if "TLG Fee" in df.columns:
        # Use existing TLG Fee column
        req.append("TLG Fee")
        _coerce_numeric(df, req)
        df["expected_cogs"] = df["GMV Net VAT"] - df["TLG Fee"]
    elif "recalc_%TLG FEE" in df.columns and "GMV Net VAT" in df.columns:
        # Calculate TLG Fee from recalc_%TLG FEE (decimal format)
        _coerce_numeric(df, req + ["recalc_%TLG FEE"])
        df["calculated_tlg_fee"] = (df["GMV Net VAT"] * df["recalc_%TLG FEE"]).round(2)
        df["expected_cogs"] = df["GMV Net VAT"] - df["calculated_tlg_fee"]
    elif "% TLG FEE" in df.columns and "GMV Net VAT" in df.columns:
        # Calculate TLG Fee from % TLG FEE (percentage format)
        _coerce_numeric(df, req + ["% TLG FEE"])
        df["calculated_tlg_fee"] = (df["GMV Net VAT"] * (df["% TLG FEE"] / 100)).round(2)
        df["expected_cogs"] = df["GMV Net VAT"] - df["calculated_tlg_fee"]
    else:
        # No way to calculate TLG Fee, return empty DataFrame
        return pd.DataFrame()

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
    # Only perform check for GB/US shipments; otherwise skip
    if "DDP Services" not in df.columns:
        return pd.DataFrame()

    # Need a tax column and Shipping Country to scope the check
    tax_col = None
    if "% Tax" in df.columns:
        tax_col = "% Tax"
    elif "VAT%" in df.columns:
        tax_col = "VAT%"
    else:
        return pd.DataFrame()

    if "Shipping Country" not in df.columns:
        return pd.DataFrame()

    # Scope to GB/US shipments only
    ship_norm = df["Shipping Country"].astype(str).str.strip().str.upper()
    in_scope = ship_norm.isin(["GB", "US"])
    if not in_scope.any():
        return pd.DataFrame()

    req = [tax_col, "DDP Services"]
    _coerce_numeric(df, req)
    mismatches = df[in_scope & (df[tax_col] > tol) & (df["DDP Services"] > tol)].copy()
    return mismatches


def sanity_check_gmv_eur(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    # Check if all required columns exist - if not, return empty DataFrame
    req = ["Sell Price", "Discount", "DDP Services", "GMV EUR", "Exchange rate"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return pd.DataFrame()
    
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
    # Ferrari-specific formula
    try:
        brand_key = st.session_state.get("brand")
    except Exception:
        brand_key = None
    if isinstance(brand_key, str) and brand_key.strip().upper() == "FE":
        # Require needed columns
        req = ["Original Price Value", "Discount Value", "GMV Net VAT", "Exchange rate"]
        # VAT may be in either column
        tax_col = "% Tax" if "% Tax" in df.columns else "VAT%" if "VAT%" in df.columns else None
        if tax_col:
            req.append(tax_col)
        # Duties/GB distinction via DDP Services; if present and > 0 then use duties>0 rule
        duties_col_present = "DDP Services" in df.columns
        missing = [c for c in req if c not in df.columns]
        if missing:
            return pd.DataFrame()
        _coerce_numeric(df, req + (["DDP Services"] if duties_col_present else []))

        base = (df["Original Price Value"].fillna(0) - df["Discount Value"].fillna(0))
        exch = df["Exchange rate"].replace(0, pd.NA)
        if tax_col:
            percent = df[tax_col].fillna(0)
        else:
            percent = pd.Series(0, index=df.index)

        # Precompute ROW expected (ignores duties): (OPV - Discount)/(1+VAT/100)
        expected_row_pre = (base / (1 + percent / 100))

        # Determine GB vs ROW using Shipping Country
        ship_series = df.get("Shipping Country")
        if ship_series is not None:
            ship_norm = ship_series.astype(str).str.strip().str.upper()
            is_gb = ship_norm == "GB"
        else:
            # If shipping country missing, default to ROW behavior
            is_gb = pd.Series([False] * len(df), index=df.index)

        # Within GB: if duties > 0 then skip VAT division; else behave like ROW
        if duties_col_present:
            duties = df["DDP Services"].fillna(0)
            expected_pre = np.where(is_gb, np.where(duties > 0, base, expected_row_pre), expected_row_pre)
        else:
            expected_pre = expected_row_pre

        # Finally divide by currency factor (exchange rate)
        df["expected_gmv_net"] = pd.Series(expected_pre, index=df.index) / exch

        diff = (df["GMV Net VAT"] - df["expected_gmv_net"]).abs()
        rel_ok = diff <= (df["expected_gmv_net"].abs() * rtol)
        abs_ok = diff <= atol
        df["gmv_net_match"] = (rel_ok.fillna(False) | abs_ok.fillna(False))
        mismatches = df.loc[~df["gmv_net_match"]].copy()
        mismatches["delta"] = mismatches["GMV Net VAT"] - mismatches["expected_gmv_net"]
        return mismatches

    # Default formula (non-FE)
    # Check for either % Tax or VAT% column
    tax_col = None
    if "% Tax" in df.columns:
        tax_col = "% Tax"
    elif "VAT%" in df.columns:
        tax_col = "VAT%"
    else:
        return pd.DataFrame()
    req = [tax_col, "GMV EUR", "GMV Net VAT"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return pd.DataFrame()
    _coerce_numeric(df, req)
    percent = df[tax_col].fillna(0)
    # HE brand: % Tax was divided by 100 on load → already decimal
    try:
        brand_key = st.session_state.get("brand")
    except Exception:
        brand_key = None
    he_decimal_mode = isinstance(brand_key, str) and brand_key.strip().upper() == "HE"
    if not he_decimal_mode:
        percent = percent
    else:
        # already decimal, no conversion
        pass
    # If HE: percent is decimal; otherwise convert from percentage
    divisor = (1 + (percent if he_decimal_mode else (percent / 100)))
    df["expected_gmv_net"] = df["GMV EUR"] / divisor
    diff = (df["GMV Net VAT"] - df["expected_gmv_net"]).abs()
    rel_ok = diff <= (df["expected_gmv_net"].abs() * rtol)
    abs_ok = diff <= atol
    df["gmv_net_match"] = (rel_ok.fillna(False) | abs_ok.fillna(False))
    mismatches = df.loc[~df["gmv_net_match"]].copy()
    mismatches["delta"] = mismatches["GMV Net VAT"] - mismatches["expected_gmv_net"]
    return mismatches

def sanity_check_original_price_value(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    """Check Original Price Value ≈ Gross price - Sales Tax - DDP Services.

    If `Gross price` is missing or null for a row, expected value is 0.
    Returns a DataFrame of mismatches with columns including the expected value and delta.
    If required columns are missing, returns an empty DataFrame.
    """
    # Required columns (Gross price optional per-row but required as a column)
    if "Gross price" not in df.columns:
        return pd.DataFrame()

    req_core = ["Original Price Value", "Sales Tax", "DDP Services"]
    missing = [c for c in req_core if c not in df.columns]
    if missing:
        return pd.DataFrame()
    # Coerce numeric
    _coerce_numeric(df, req_core + ["Gross price"])

    # Build expected with rule: if Gross price is NaN -> expected 0; else gross - taxes - ddp
    base = pd.to_numeric(df["Gross price"], errors="coerce")
    sales_tax = pd.to_numeric(df["Sales Tax"], errors="coerce").fillna(0)
    ddp = pd.to_numeric(df["DDP Services"], errors="coerce").fillna(0)

    expected = (base - sales_tax - ddp)
    expected = expected.where(base.notna(), 0)

    df_work = df.copy()
    df_work["expected_original_price_value"] = expected

    # Compare using tolerances
    diff = (df_work["Original Price Value"] - df_work["expected_original_price_value"]).abs()
    rel_ok = diff <= (df_work["expected_original_price_value"].abs() * rtol).fillna(0)
    abs_ok = diff <= atol
    df_work["opv_match"] = rel_ok | abs_ok

    mismatches = df_work.loc[~df_work["opv_match"]].copy()
    mismatches["delta"] = mismatches["Original Price Value"] - mismatches["expected_original_price_value"]
    return mismatches

def fetch_orders_returns(
    start_date: str,
    end_date: str,
    brand: str,
    source_company: str,
    order_numbers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch orders/returns with two modes:

    - Default: date window by row_type (O uses invoice_creation_date, R uses credit_note_creation_date)
    - If order_numbers provided (list), ignore dates and fetch rows by order_number instead.
    """
    is_fe = str(brand).strip().upper() == "FE"
    if order_numbers and len(order_numbers) > 0:
        # Normalize to strings for BQ
        ids = [str(x).strip().upper() for x in order_numbers if str(x).strip()]
        if not ids:
            return pd.DataFrame()
        sql = f"""
            SELECT *
            FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
            WHERE brand = @brand
              {'' if is_fe else 'AND source_company = @source_company'}
              AND UPPER(CAST(order_number AS STRING)) IN UNNEST(@order_numbers)
        """
        params = [
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
            bigquery.ArrayQueryParameter("order_numbers", "STRING", ids),
        ]
        if not is_fe:
            params.insert(1, bigquery.ScalarQueryParameter("source_company", "STRING", source_company))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return client.query(sql, job_config=job_config).to_dataframe()
    else:
        sql = f"""
            SELECT *
            FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
            WHERE brand = @brand
              {'' if is_fe else 'AND source_company = @source_company'}
              AND (
                (row_type = 'O' AND DATE(invoice_creation_date) BETWEEN @start_date AND @end_date)
                OR
                (row_type = 'R' AND DATE(credit_note_creation_date) BETWEEN @start_date AND @end_date)
              )
        """
        params = [
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
        if not is_fe:
            params.append(bigquery.ScalarQueryParameter("source_company", "STRING", source_company))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
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


def fetch_items_rc_brand_parts(product_ids: List[str]) -> pd.DataFrame:
    """
    Fetch `product_id`, `model`, `variant`, `fabric`, `color` from `bi.erp_items` for provided IDs.
    """
    if not product_ids:
        return pd.DataFrame(columns=["product_id", "model", "variant", "fabric", "color"])
    sql = f"""
        SELECT DISTINCT
          CAST(product_id AS STRING) AS product_id,
          CAST(model       AS STRING) AS model,
          CAST(variant     AS STRING) AS variant,
          CAST(fabric      AS STRING) AS fabric,
          CAST(color       AS STRING) AS color
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
          currency,
          reset_date
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
    - Thresholds: if present, pick row whose [min,max] contains the selected base
      (threshold_base or fee_calc_base). If none match, fall back to no-threshold
      rows with a defined fee.
    - Writes only `recalc_%TLG FEE`. Does not modify existing fee columns.

    Notes / Fixes vs. previous version:
    - Do NOT cast ARRAY columns to strings; keep them as lists from BigQuery.
    - `_parse_list_field` is list-aware and tolerant of string forms (in case upstream changes).
    - Excludes ARRAY columns from generic string normalization.
    """

    if df is None or df.empty or fee_config_df is None or fee_config_df.empty:
        # Nothing to do; return original df unchanged
        return df

    # Work on copies to avoid mutating caller inputs
    working = fee_config_df.copy(deep=True)
    out = df.copy(deep=True)
    # -----------------------------
    # Helpers
    # -----------------------------
    def _nullify(series: pd.Series) -> pd.Series:
        """Treat common null-like sentinels as missing."""
        return series.replace(
            {"": pd.NA, "null": pd.NA, "None": pd.NA, "none": pd.NA, "NaN": pd.NA, "nan": pd.NA}
        )

    def _normalize_yes_no_flag(v: Any) -> Optional[int]:
        """Map truthy/falsey flags to 1/0; return None if not parseable."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip().upper()
        if s in {"1", "Y", "YES", "TRUE", "T"}:
            return 1
        if s in {"0", "N", "NO", "FALSE", "F"}:
            return 0
        return None

    def _as_upper_list(xs: Iterable[Any]) -> list[str]:
        return [str(x).strip().upper() for x in xs if str(x).strip()]

    def _parse_list_field(x: Any) -> list[str]:
        """
        Parse comma-separated values only.
        - Accept list/tuple by normalizing and uppercasing
        - Accept strings like "A,B,C" or "[A,B,C]" (optional brackets)
        - If '*' appears anywhere in the string, treat as wildcard ['*']
        - Returns [] for blank/NA
        """
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []

        if isinstance(x, (list, tuple)):
            vals = _as_upper_list(x)
            return ["*"] if any(v == "*" for v in vals) else vals

        # Numpy/pandas array-like: use tolist()
        if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
            try:
                seq = x.tolist()
                if not isinstance(seq, (list, tuple)):
                    seq = [seq]
                vals = _as_upper_list(seq)
                return ["*"] if any(v == "*" for v in vals) else vals
            except Exception:
                pass

        s = str(x).strip()
        if not s:
            return []
        if "*" in s:
            return ["*"]
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            s = s[1:-1].strip()
        parts = [p.strip().strip("'\"") for p in s.split(",")]
        parts = [p for p in parts if p]
        return [p.upper() for p in parts]


    def _erp_ok(cfg_val: Any, ent: Optional[str]) -> bool:
        if ent is None or (isinstance(ent, float) and pd.isna(ent)) or str(ent).strip() == "":
            return True
        if pd.isna(cfg_val):
            return True
        s = str(cfg_val).strip()
        return s == "*" or s.upper() == str(ent).strip().upper()

    def _brand_ok(cfg_val: Any, b: str) -> bool:
        if pd.isna(cfg_val):
            return True
        s = str(cfg_val).strip()
        return s == "*" or s.upper() == str(b).strip().upper()

    def _loc_ok(cfg_val: Any, omsv: Optional[str]) -> bool:
        # Must have an OMSV to match against
        if omsv is None or (isinstance(omsv, float) and pd.isna(omsv)) or str(omsv).strip() == "":
            return False
        raw = "" if cfg_val is None or (isinstance(cfg_val, float) and pd.isna(cfg_val)) else str(cfg_val)
        if not raw:
            # Empty config means no restriction
            return True
        # Wildcard allows all
        if "*" in raw:
            return True
        # SQL LIKE-style substring match (case-insensitive)
        return str(omsv).strip().upper() in raw.strip().upper()

    def _ship_match(incl_list: list[str], excl_list: list[str], ship_val: str) -> bool:
        """Inclusion beats exclusion; '*' in either list is wildcard (no restriction)."""
        incl = list(incl_list) if isinstance(incl_list, (list, tuple)) else []
        excl = list(excl_list) if isinstance(excl_list, (list, tuple)) else []
        ship = (ship_val or "").strip().upper()

        # Wildcard in inclusion => no restriction
        if incl and incl == ["*"]:
            incl = []

        # Wildcard in exclusion => no restriction
        if excl and excl == ["*"]:
            excl = []

        if incl:
            if ship not in set(incl):
                return False
        if excl:
            if ship in set(excl):
                return False
        return True

    # -----------------------------
    # Normalize config
    # -----------------------------

    # 1) String-ish columns (EXCLUDE ARRAY columns so we don't coerce lists to strings)
    stringish_cols = [
        "brand", "erp_entity", "fee_calc_base", "threshold_base", "endless_aisle"
    ]
    for col in stringish_cols:
        if col in working.columns:
            working[col] = working[col].astype("string").str.strip()

    # 2) Numeric columns
    for col in ["fee_perc", "min_threshold_ytd", "max_threshold_ytd"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    # 3) Null-like to NA for stringish matching columns
    for col in ["brand", "erp_entity", "endless_aisle"]:
        if col in working.columns:
            working[col] = _nullify(working[col])


    # 4) Inclusion/Exclusion ARRAY columns: keep lists if they already are; parse if strings
    if "shipping_country_inclusion" in working.columns:
        working["_incl_list"] = working["shipping_country_inclusion"].apply(_parse_list_field)
    else:
        working["_incl_list"] = [[] for _ in range(len(working))]

    if "shipping_country_exclusion" in working.columns:
        working["_excl_list"] = working["shipping_country_exclusion"].apply(_parse_list_field)
    else:
        working["_excl_list"] = [[] for _ in range(len(working))]
    

    # -----------------------------
    # Static YTD bases
    # -----------------------------
    ytd_gmv = float(static_ytd_gmv or 0.0)
    ytd_nmv = float(static_ytd_nmv or 0.0)


    def _threshold_ok(cfg_row: pd.Series) -> bool:
        min_thr = cfg_row.get("min_threshold_ytd")
        max_thr = cfg_row.get("max_threshold_ytd")
        base = str(cfg_row.get("threshold_base")).strip().upper()
        if pd.isna(min_thr) and pd.isna(max_thr):
            return False
        value = ytd_gmv if base == "GMV" else ytd_nmv if base == "NMV" else ytd_gmv
        try:
            low = float(min_thr) if not pd.isna(min_thr) else float("-inf")
            high = float(max_thr) if not pd.isna(max_thr) else float("inf")
            return float(value) >= low and float(value) <= high
        except Exception:
            return False

    # -----------------------------
    # Prepare transaction rows
    # -----------------------------
    out["_ship"] = (
        out.get("Shipping Country", pd.Series(index=out.index))
        .astype("string").str.strip().str.upper()
    )

    if "OMS Location Name" in out.columns:
        out["_oms"] = out["OMS Location Name"].astype("string").str.strip().str.upper()
    else:
        out["_oms"] = pd.Series([None] * len(out), index=out.index)

    cv_brand = (str(brand).strip().upper() == "CV")
    if cv_brand and "Endless Aisle" in out.columns:
        out["_ea"] = out["Endless Aisle"].apply(_normalize_yes_no_flag)
    else:
        out["_ea"] = pd.Series([None] * len(out), index=out.index)

    # -----------------------------
    # Row-wise matching & fee selection
    # -----------------------------
    # Initialize target column so we can assign per-row safely
    out["recalc_%TLG FEE"] = pd.Series([pd.NA] * len(out), index=out.index)

    # for _, row in out.iterrows():
    for _, row in out.iterrows():
        try:
            ship = row.get("_ship") or ""
            omsv = row.get("_oms")
            endless_val = row.get("_ea")

            cand = working

            # ERP entity
            if "erp_entity" in cand.columns:
                cand = cand[cand["erp_entity"].apply(lambda x: _erp_ok(x, erp_entity))]

            # Location
            if "location_code" in cand.columns and omsv:
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
            selected_row = None
            if not cand.empty:
                # Prefer matching threshold rows
                with_thr = cand[cand.apply(_threshold_ok, axis=1)].copy()
                if not with_thr.empty:
                    with_thr["_low"] = with_thr["min_threshold_ytd"].fillna(0.0)
                    # deterministic tie-break: lowest min_threshold then highest fee
                    with_thr = with_thr.sort_values(by=["_low", "fee_perc"], ascending=[True, False])
                    selected_row = with_thr.iloc[0]
                    fee = selected_row.get("fee_perc")

                else:
                    # Fallback: rows without thresholds (or any with a fee)
                    no_thr = cand[(cand["min_threshold_ytd"].isna()) & (cand["max_threshold_ytd"].isna())]
                    pool = no_thr if not no_thr.empty else cand
                    pool = pool.dropna(subset=["fee_perc"]).copy()
                    if not pool.empty:
                        pool = pool.sort_values(by=["fee_perc"], ascending=[False])
                        selected_row = pool.iloc[0]
                        fee = selected_row.get("fee_perc")

            # -----------------------------
            # Apply GMV/NMV base & returns rules
            # -----------------------------
            applied_fee = fee

            # Detect returns robustly
            try:
                row_type_val = str(row.get("Row Type", "")).strip().upper()
            except Exception:
                row_type_val = ""
            try:
                type_text = str(row.get("Type", "")).strip().upper()
            except Exception:
                type_text = ""
            qty_val = pd.to_numeric(row.get("Qty"), errors="coerce") if "Qty" in row.index else pd.NA

            is_return = False
            if cv_brand and type_text == "RETURN":
                is_return = True
            elif row_type_val == "R":
                is_return = True
            elif any(k in type_text for k in ["CREDIT", "CREDITO", "NOTA"]):
                # Handle 'Credit Note' / 'Nota di Credito'
                is_return = True
            elif pd.notna(qty_val) and float(qty_val) < 0:
                is_return = True

            # Determine calc base once
            if selected_row is not None:
                calc_base = str(selected_row.get("fee_calc_base") or "").strip().upper()
            else:
                calc_base = ""

            if is_return:
                # Returns: apply fee only when base is NMV; zero out for GMV
                if calc_base == "GMV":
                    applied_fee = 0.0 if applied_fee is not None else 0.0
                else:
                    # NMV (or unknown): keep applied_fee as-is so returns are charged too
                    pass
            else:
                # Orders: for NMV, apply gating on Qualità Reso. (fee only when equals 0)
                if calc_base == "NMV":
                    qreso = pd.to_numeric(row.get("Qualità Reso."), errors="coerce") if "Qualità Reso." in row.index else 0.0
                    if pd.notna(qreso) and float(qreso) != 0.0:
                        applied_fee = 0.0 if applied_fee is not None else 0.0

            value = float(applied_fee) if applied_fee is not None and not pd.isna(applied_fee) else None
            out.at[row.name, "recalc_%TLG FEE"] = value
        except Exception as e:
            print("apply_tlg_fee row error", row.name, type(e).__name__, e)
    # Cleanup temp cols
    out = out.drop(columns=[c for c in ["_ship", "_oms", "_ea"] if c in out.columns], errors="ignore")
    print(out)
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


def fetch_reset_aware_totals(
    brand: str,
    period_start_date: str,
    period_end_date: str,
) -> Tuple[float, float]:
    """Compute GMV/NMV totals from the brand reset_date month/day up to period_start_date.

    Rules:
    - Use `config.brand_fee.reset_date` month and day, ignore its year.
    - Determine the most recent reset boundary on/before period_start_date.
    - Sum orders/returns between that boundary (inclusive) and the day before period_start_date.
    - Return (gmv_sum, nmv_sum) as floats; if no rows, (0.0, 0.0).
    - If reset_date missing, fall back to calendar YTD logic using Jan 1 of period_start_date year.
    """
    # 1) Fetch reset_date (could be multiple config rows; take first non-null)
    cfg = fetch_tlg_fee_config(brand)
    reset_ts: Optional[pd.Timestamp] = None
    if cfg is not None and not cfg.empty and "reset_date" in cfg.columns:
        try:
            # Normalize to pandas Timestamp; pick first non-null
            col = pd.to_datetime(cfg["reset_date"], errors="coerce")
            col = col.dropna()
            if not col.empty:
                reset_ts = pd.to_datetime(col.iloc[0])
        except Exception:
            reset_ts = None

    # 2) Parse input dates
    start_dt = pd.to_datetime(period_start_date).normalize()
    end_dt = pd.to_datetime(period_end_date).normalize()

    # Guard: if start after end, clamp
    if start_dt > end_dt:
        end_dt = start_dt

    # 3) Determine reset boundary date (inclusive window start)
    if reset_ts is not None:
        reset_month = int(reset_ts.month)
        reset_day = int(reset_ts.day)
        # Construct the reset date in the start year
        candidate = pd.Timestamp(year=start_dt.year, month=reset_month, day=reset_day)
        if candidate > start_dt:
            # If the reset in the same year is after the period start, use previous year
            candidate = pd.Timestamp(year=start_dt.year - 1, month=reset_month, day=reset_day)
        window_start = candidate
    else:
        # Fallback to Jan 1 of the period start year
        window_start = pd.Timestamp(year=start_dt.year, month=1, day=1)

    # We sum up to the day before period_start_date
    window_end = start_dt - pd.Timedelta(days=1)

    # If the computed window is empty or negative, return zeros
    if window_end < window_start:
        return 0.0, 0.0

    print(window_start, window_end)

    # 4) Query BI for sums in the window
    sql = f"""
        SELECT
          ROUND(SUM(CASE WHEN row_type = 'O' THEN gmv_eur ELSE 0 END), 2) AS total_gmv_eur,
          ROUND(SUM(CASE WHEN row_type = 'R' THEN rmv_eur ELSE 0 END), 2) AS total_returns_eur,
          ROUND(
            SUM(CASE WHEN row_type = 'O' THEN gmv_eur ELSE 0 END)
            + SUM(CASE WHEN row_type = 'R' THEN rmv_eur ELSE 0 END),
          2) AS total_nmv_eur
        FROM `{BQ_PROJECT}.bi.orders_returns_new`
        WHERE brand = @brand
          AND DATE(CASE
                WHEN row_type = 'O' THEN invoice_creation_date
                WHEN row_type = 'R' THEN credit_note_creation_date
              END) BETWEEN @window_start AND @window_end
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand.strip().upper()),
            bigquery.ScalarQueryParameter("window_start", "DATE", window_start.strftime("%Y-%m-%d")),
            bigquery.ScalarQueryParameter("window_end", "DATE", window_end.strftime("%Y-%m-%d")),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    if df is None or df.empty:
        return 0.0, 0.0
    gmv = float(pd.to_numeric(df.iloc[0].get("total_gmv_eur"), errors="coerce") or 0.0)
    nmv = float(pd.to_numeric(df.iloc[0].get("total_nmv_eur"), errors="coerce") or 0.0)
    return gmv, nmv
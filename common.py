# common.py
import os
import io
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import pandas as pd
from google.cloud import bigquery

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


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_invoicing_report(file: Union[io.BytesIO, str]) -> pd.DataFrame:
    import csv
    headers = [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.", "Matricolari", "__extra__"
    ]
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

    # Parse dates (EU format)
    for dcol in ["Export Date", "Date", "Data Fattura"]:
        if dcol in df:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)

    num_cols = [
        "Qty", "COGS", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2"
    ]
    df = _coerce_numeric(df, num_cols)
    return df

def sanity_check_tlg_fee(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    req = ["TLG Fee", "GMV Net VAT", "% TLG FEE"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    # Avoid division by zero and NaN
    percent = df["% TLG FEE"].replace(0, pd.NA)
    df["expected_tlg_fee"] = (df["GMV Net VAT"] * (percent / 100)).round(2)
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

def fetch_orders_returns(start_date: str, end_date: str, brand: str) -> pd.DataFrame:
    client = bigquery.Client(project=BQ_PROJECT)
    sql = f"""
        SELECT *
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE brand = @brand AND DATE(order_date) BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()

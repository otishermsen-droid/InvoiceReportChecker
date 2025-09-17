# common.py
import os
import io
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

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


def load_invoicing_report(file: io.BytesIO | str) -> pd.DataFrame:
    import csv
    headers = [
        "StoreID", "Export Date", "Transaction ID", "Date", "Row ID",
        "Row Type", "EAN", "Product ID", "Color", "Size", "Qty", "COGS", "Season", "Order Number",
        "Type", "Numero Fattura / Nota di Credito", "Data Fattura", "Shipping Country",
        "Currency", "% Tax", "Original Price", "Sell Price", "Discount", "DDP Services",
        "Exchange rate", "GMV EUR", "GMV Net VAT", "% TLG FEE", "TLG Fee", "COGS2",
        "Qualità Reso.","Matricolari","__extra__"
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
        "Qty","COGS","% Tax","Original Price","Sell Price","Discount","DDP Services",
        "Exchange rate","GMV EUR","GMV Net VAT","% TLG FEE","TLG Fee","COGS2"
    ]
    df = _coerce_numeric(df, num_cols)
    return df


def sanity_check_cogs(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    req = ["COGS", "GMV Net VAT", "TLG Fee", "DDP Services"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    df["expected_cogs"] = df["GMV Net VAT"] - df["TLG Fee"] - df["DDP Services"]
    diff = (df["COGS"] - df["expected_cogs"]).abs()
    rel_ok = diff <= (df["expected_cogs"].abs() * rtol).fillna(0)
    abs_ok = diff <= atol
    df["cogs_match"] = rel_ok | abs_ok
    mismatches = df.loc[~df["cogs_match"]].copy()
    mismatches["delta"] = df["COGS"] - df["expected_cogs"]
    return mismatches


def auto_fix_cogs(
    df: pd.DataFrame, mismatches: pd.DataFrame | None = None
) -> Tuple[pd.DataFrame, int]:
    """Fix COGS and COGS2 values for the provided mismatches.

    Parameters
    ----------
    df:
        The full invoicing dataframe.
    mismatches:
        Optional dataframe returned by :func:`sanity_check_cogs`. When omitted the
        function recomputes the mismatches internally.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A copy of the input dataframe with corrected values and the number of
        rows updated.
    """

    if mismatches is None:
        mismatches = sanity_check_cogs(df.copy())

    if mismatches.empty:
        return df.copy(), 0

    updated_df = df.copy()
    index_to_fix = mismatches.index.intersection(updated_df.index)
    if len(index_to_fix) == 0:
        return updated_df, 0

    expected = mismatches.loc[index_to_fix, "expected_cogs"]
    updated_df.loc[index_to_fix, "COGS"] = expected

    if "COGS2" in updated_df.columns:
        updated_df.loc[index_to_fix, "COGS2"] = expected

    logging.info("Auto-fixed COGS for %d rows.", len(index_to_fix))
    return updated_df, len(index_to_fix)


def sanity_checks_ddp_tax(df: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    req = ["% Tax", "DDP Services"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    mismatches = df[(df["% Tax"] > tol) & (df["DDP Services"] > tol)].copy()
    return mismatches


def sanity_check_gmv(df: pd.DataFrame, atol: float = 0.01, rtol: float = 0.01) -> pd.DataFrame:
    req = ["Sell Price", "Discount", "DDP Services", "GMV EUR"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    _coerce_numeric(df, req)
    df["expected_gmv"] = df["Sell Price"] - df["Discount"] - df["DDP Services"]
    diff = (df["GMV EUR"] - df["expected_gmv"]).abs()
    rel_ok = diff <= (df["expected_gmv"].abs() * rtol).fillna(0)
    abs_ok = diff <= atol
    df["gmv_match"] = rel_ok | abs_ok
    mismatches = df.loc[~df["gmv_match"]].copy()
    mismatches["delta"] = df["GMV EUR"] - df["expected_gmv"]
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

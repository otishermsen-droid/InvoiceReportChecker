import streamlit as st
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

from common import client, get_supported_brands

TABLE_FQN = "tlg-business-intelligence-prd.config.invoice_checker_duties_config"

st.set_page_config(page_title="DDP Duties Config", layout="wide")
st.title("⚙️ DDP Duties Configuration")

st.markdown("""
<style>
[data-testid="baseButton-primary"] {
    background-color: #28a745 !important;
    color: white !important;
    border: none !important;
}
[data-testid="baseButton-primary"]:hover {
    background-color: #218838 !important;
    color: white !important;
}
.stButton > button[kind="primary"] {
    background-color: #28a745 !important;
    color: white !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #218838 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

ALL_COLUMNS = [
    "row_key",
    "brand_code",
    "shipping_country",
    "oms_location_name",
    "duty_recalculation",
    "ddp_fix",
    "ddp_perc",
    "treshold_amount",
    "currency_code",
    "application_date",
    "origin_country",
    "source_company",
]


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=ALL_COLUMNS)
    # Ensure all expected columns exist
    for c in ALL_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    # Normalize row_key as trimmed string (do not change case; GUID semantics)
    if "row_key" in out.columns:
        out["row_key"] = out["row_key"].astype(str).str.strip().where(out["row_key"].notna(), None)

    # String columns normalized to uppercase trimmed
    for c in ["brand_code", "shipping_country", "oms_location_name", "currency_code", "origin_country", "source_company"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.upper().where(out[c].notna(), None)

    # Numeric columns
    for c in ["duty_recalculation"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    for c in ["ddp_fix", "ddp_perc", "treshold_amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Dates as YYYY-MM-DD string
    if "application_date" in out.columns:
        def _fmt(x):
            if pd.isna(x) or x in ("", None):
                return ""
            try:
                return pd.to_datetime(x).strftime("%Y-%m-%d")
            except Exception:
                return ""
        out["application_date"] = out["application_date"].apply(_fmt)

    out = out[ALL_COLUMNS]
    return out


def row_key_tuple(row: pd.Series) -> Tuple:
    # Prefer explicit row_key when provided
    rk = str(row.get("row_key") or "").strip()
    if rk:
        return (rk,)
    # Fallback: use full-row composite (excluding row_key)
    return tuple(row.get(c) for c in ALL_COLUMNS if c != "row_key")


# Mapping of 2-letter brand -> numeric brand_code (subset: only brands we support)
BRAND_TO_CODE: Dict[str, str] = {
    "AL": "54",
    "AT": "24",
    "BO": "45",
    "CA": "05",
    "CV": "50",
    "FE": "48",
    "FO": "34",
    "HB": "40",
    "HE": "33",
    "MA": "52",
    "MO": "41",
    "PJ": "35",
    "PL": "49",
    "RC": "46",
    "FA": "47",
    "FU": "37",
}


def resolve_brand_code(brand_two_letter: Optional[str]) -> Optional[str]:
    if not isinstance(brand_two_letter, str):
        return None
    key = brand_two_letter.strip().upper()
    return BRAND_TO_CODE.get(key)


def fetch_config_df(brand_code: Optional[str]) -> pd.DataFrame:
    where = []
    params = []
    if isinstance(brand_code, str) and brand_code.strip():
        where.append("UPPER(TRIM(brand_code)) = UPPER(@brand_code)")
        params.append(("brand_code", brand_code.strip().upper()))

    # Try selecting with row_key; if column doesn't exist in BQ, fallback without it
    from google.cloud import bigquery
    cols_with_rowkey = list(ALL_COLUMNS)
    cols_without_rowkey = [c for c in ALL_COLUMNS if c != "row_key"]

    def _run_query(select_cols: List[str]):
        sql = f"SELECT {', '.join(select_cols)} FROM `{TABLE_FQN}`"
        if where:
            sql += " WHERE " + " AND ".join(where)
        job_config = None
        if params:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params]
            )
        return client.query(sql, job_config=job_config).to_dataframe()

    try:
        df = _run_query(cols_with_rowkey)
    except Exception:
        df = _run_query(cols_without_rowkey)
    return canonicalize_df(df)


def validate_df(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    # Required brand_code
    if "brand_code" in df.columns and df["brand_code"].astype(str).str.strip().eq("").any():
        errors.append("brand_code is required on all rows.")
    # duty_recalculation numeric (0/1 typical)
    if "duty_recalculation" in df.columns:
        dr = pd.to_numeric(df["duty_recalculation"], errors="coerce")
        if dr.isna().any():
            errors.append("duty_recalculation must be an integer (0/1).")
    # Monetary/percent fields numeric
    for c in ["ddp_fix", "ddp_perc", "treshold_amount"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.isna().any():
                errors.append(f"{c} must be numeric.")
    # application_date valid (only when non-empty)
    if "application_date" in df.columns:
        s = df["application_date"].astype(str).str.strip()
        non_empty = s != ""
        parsed = pd.to_datetime(s.where(non_empty, pd.NA), errors="coerce")
        invalid_mask = non_empty & parsed.isna()
        if invalid_mask.any():
            errors.append("application_date must be a valid date (YYYY-MM-DD).")
    # Composite/row_key uniqueness
    keys = df.apply(row_key_tuple, axis=1)
    if keys.duplicated().any():
        errors.append("Composite key must be unique. Found duplicate key(s) in the edited data.")
    return errors


def diff_configs(original: pd.DataFrame, edited: pd.DataFrame):
    # Returns (to_insert, to_update, to_delete)
    orig_canon = canonicalize_df(original)
    edit_canon = canonicalize_df(edited)

    orig_map: Dict[Tuple, pd.Series] = {row_key_tuple(r): r for _, r in orig_canon.iterrows()}
    edit_map: Dict[Tuple, pd.Series] = {row_key_tuple(r): r for _, r in edit_canon.iterrows()}

    orig_keys = set(orig_map.keys())
    edit_keys = set(edit_map.keys())

    to_delete_keys = sorted(list(orig_keys - edit_keys))
    to_insert_keys = sorted(list(edit_keys - orig_keys))
    common_keys = orig_keys & edit_keys

    to_delete = [orig_map[k] for k in to_delete_keys]
    to_insert = [edit_map[k] for k in to_insert_keys]
    to_update: List[pd.Series] = []

    # Detect updates on shared keys. If matched by row_key (len==1), compare all columns except row_key.
    # If matched by composite fallback, key includes all fields so there will be no changes.
    for k in common_keys:
        o = orig_map[k]
        e = edit_map[k]
        changed = False
        # Compare every column except row_key
        for c in [c for c in ALL_COLUMNS if c != "row_key"]:
            ov = o.get(c)
            ev = e.get(c)
            # Treat NaN==NaN as equal
            if (pd.isna(ov) and pd.isna(ev)) or (ov == ev):
                continue
            changed = True
            break
        if changed:
            to_update.append(e)

    return to_insert, to_update, to_delete


def delete_rows(rows: List[pd.Series]) -> int:
    from google.cloud import bigquery
    total = 0
    for r in rows:
        row_key = (str(r.get("row_key") or "").strip() or None)
        sql = f"""
DELETE FROM `{TABLE_FQN}`
WHERE
  (
    @has_row_key = TRUE AND row_key = @row_key
  )
  OR
  (
    @has_row_key = FALSE AND
    ((brand_code IS NULL AND @brand_code IS NULL) OR UPPER(brand_code) = UPPER(@brand_code)) AND
    ((shipping_country IS NULL AND @shipping_country IS NULL) OR UPPER(shipping_country) = UPPER(@shipping_country)) AND
    ((oms_location_name IS NULL AND @oms_location_name IS NULL) OR UPPER(oms_location_name) = UPPER(@oms_location_name)) AND
    (SAFE_CAST(duty_recalculation AS INT64) IS NOT DISTINCT FROM @duty_recalculation) AND
    (SAFE_CAST(ddp_fix AS NUMERIC) IS NOT DISTINCT FROM @ddp_fix) AND
    (SAFE_CAST(ddp_perc AS NUMERIC) IS NOT DISTINCT FROM @ddp_perc) AND
    (SAFE_CAST(treshold_amount AS NUMERIC) IS NOT DISTINCT FROM @treshold_amount) AND
    ((currency_code IS NULL AND @currency_code IS NULL) OR UPPER(currency_code) = UPPER(@currency_code)) AND
    (application_date IS NOT DISTINCT FROM @application_date) AND
    ((origin_country IS NULL AND @origin_country IS NULL) OR UPPER(origin_country) = UPPER(@origin_country)) AND
    ((source_company IS NULL AND @source_company IS NULL) OR UPPER(source_company) = UPPER(@source_company))
  )
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("has_row_key", "BOOL", row_key is not None),
                bigquery.ScalarQueryParameter("row_key", "STRING", row_key),
                bigquery.ScalarQueryParameter("brand_code", "STRING", r.get("brand_code")),
                bigquery.ScalarQueryParameter("shipping_country", "STRING", r.get("shipping_country")),
                bigquery.ScalarQueryParameter("oms_location_name", "STRING", r.get("oms_location_name")),
                bigquery.ScalarQueryParameter("duty_recalculation", "INT64", None if pd.isna(r.get("duty_recalculation")) else int(r.get("duty_recalculation"))),
                bigquery.ScalarQueryParameter("ddp_fix", "NUMERIC", None if pd.isna(r.get("ddp_fix")) else float(r.get("ddp_fix"))),
                bigquery.ScalarQueryParameter("ddp_perc", "NUMERIC", None if pd.isna(r.get("ddp_perc")) else float(r.get("ddp_perc"))),
                bigquery.ScalarQueryParameter("treshold_amount", "NUMERIC", None if pd.isna(r.get("treshold_amount")) else float(r.get("treshold_amount"))),
                bigquery.ScalarQueryParameter("currency_code", "STRING", r.get("currency_code")),
                bigquery.ScalarQueryParameter("application_date", "DATE", None if (r.get("application_date") in (None, "")) else r.get("application_date")),
                bigquery.ScalarQueryParameter("origin_country", "STRING", r.get("origin_country")),
                bigquery.ScalarQueryParameter("source_company", "STRING", r.get("source_company")),
            ]
        )
        res = client.query(sql, job_config=job_config).result()
        total += res.total_rows or 0
    return total


def upsert_rows(rows: List[pd.Series]) -> int:
    from google.cloud import bigquery
    total = 0
    for r in rows:
        row_key = (str(r.get("row_key") or "").strip() or None)
        sql = f"""
MERGE `{TABLE_FQN}` T
USING (SELECT
  @row_key AS row_key,
  @brand_code AS brand_code,
  @shipping_country AS shipping_country,
  @oms_location_name AS oms_location_name,
  @duty_recalculation AS duty_recalculation,
  @ddp_fix AS ddp_fix,
  @ddp_perc AS ddp_perc,
  @treshold_amount AS treshold_amount,
  @currency_code AS currency_code,
  @application_date AS application_date,
  @origin_country AS origin_country,
  @source_company AS source_company
) S
ON (
  (S.row_key IS NOT NULL AND T.row_key = S.row_key)
  OR
  (
    S.row_key IS NULL AND
    ((T.brand_code IS NULL AND S.brand_code IS NULL) OR UPPER(T.brand_code) = UPPER(S.brand_code)) AND
    ((T.shipping_country IS NULL AND S.shipping_country IS NULL) OR UPPER(T.shipping_country) = UPPER(S.shipping_country)) AND
    ((T.oms_location_name IS NULL AND S.oms_location_name IS NULL) OR UPPER(T.oms_location_name) = UPPER(S.oms_location_name)) AND
    (SAFE_CAST(T.duty_recalculation AS INT64) IS NOT DISTINCT FROM S.duty_recalculation) AND
    (SAFE_CAST(T.ddp_fix AS NUMERIC) IS NOT DISTINCT FROM S.ddp_fix) AND
    (SAFE_CAST(T.ddp_perc AS NUMERIC) IS NOT DISTINCT FROM S.ddp_perc) AND
    (SAFE_CAST(T.treshold_amount AS NUMERIC) IS NOT DISTINCT FROM S.treshold_amount) AND
    ((T.currency_code IS NULL AND S.currency_code IS NULL) OR UPPER(T.currency_code) = UPPER(S.currency_code)) AND
    (T.application_date IS NOT DISTINCT FROM S.application_date) AND
    ((T.origin_country IS NULL AND S.origin_country IS NULL) OR UPPER(T.origin_country) = UPPER(S.origin_country)) AND
    ((T.source_company IS NULL AND S.source_company IS NULL) OR UPPER(T.source_company) = UPPER(S.source_company))
  )
)
WHEN MATCHED THEN UPDATE SET
  row_key = S.row_key,
  brand_code = S.brand_code,
  shipping_country = S.shipping_country,
  oms_location_name = S.oms_location_name,
  duty_recalculation = S.duty_recalculation,
  ddp_fix = S.ddp_fix,
  ddp_perc = S.ddp_perc,
  treshold_amount = S.treshold_amount,
  currency_code = S.currency_code,
  application_date = S.application_date,
  origin_country = S.origin_country,
  source_company = S.source_company
WHEN NOT MATCHED THEN INSERT
  (row_key, brand_code, shipping_country, oms_location_name, duty_recalculation, ddp_fix, ddp_perc, treshold_amount, currency_code, application_date, origin_country, source_company)
  VALUES
  (S.row_key, S.brand_code, S.shipping_country, S.oms_location_name, S.duty_recalculation, S.ddp_fix, S.ddp_perc, S.treshold_amount, S.currency_code, S.application_date, S.origin_country, S.source_company)
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("row_key", "STRING", row_key),
                bigquery.ScalarQueryParameter("brand_code", "STRING", r.get("brand_code")),
                bigquery.ScalarQueryParameter("shipping_country", "STRING", r.get("shipping_country")),
                bigquery.ScalarQueryParameter("oms_location_name", "STRING", r.get("oms_location_name")),
                bigquery.ScalarQueryParameter("duty_recalculation", "INT64", None if pd.isna(r.get("duty_recalculation")) else int(r.get("duty_recalculation"))),
                bigquery.ScalarQueryParameter("ddp_fix", "NUMERIC", None if pd.isna(r.get("ddp_fix")) else float(r.get("ddp_fix"))),
                bigquery.ScalarQueryParameter("ddp_perc", "NUMERIC", None if pd.isna(r.get("ddp_perc")) else float(r.get("ddp_perc"))),
                bigquery.ScalarQueryParameter("treshold_amount", "NUMERIC", None if pd.isna(r.get("treshold_amount")) else float(r.get("treshold_amount"))),
                bigquery.ScalarQueryParameter("currency_code", "STRING", r.get("currency_code")),
                bigquery.ScalarQueryParameter("application_date", "DATE", None if (r.get("application_date") in (None, "")) else r.get("application_date")),
                bigquery.ScalarQueryParameter("origin_country", "STRING", r.get("origin_country")),
                bigquery.ScalarQueryParameter("source_company", "STRING", r.get("source_company")),
            ]
        )
        client.query(sql, job_config=job_config).result()
        total += 1
    return total


# Sidebar filters
with st.sidebar:
    st.markdown("### Filter")
    brand_options = get_supported_brands()
    default_idx = 0
    pre = st.session_state.get("brand")
    if isinstance(pre, str) and pre in brand_options:
        default_idx = brand_options.index(pre)
    selected_brand = st.selectbox("Brand", options=brand_options, index=default_idx, help="2-letter brand code")
    st.session_state["brand"] = selected_brand
    mapped_code = resolve_brand_code(selected_brand)
    if mapped_code:
        st.caption(f"Mapped brand_code: {mapped_code}")
    else:
        st.caption("Mapped brand_code: (not available for this brand; showing all)")

    load_btn = st.button("Load Table", type="primary", use_container_width=True)

if "ddp_cfg_original" not in st.session_state:
    st.session_state.ddp_cfg_original = pd.DataFrame(columns=ALL_COLUMNS)
if "ddp_cfg_edited" not in st.session_state:
    st.session_state.ddp_cfg_edited = pd.DataFrame(columns=ALL_COLUMNS)

if load_btn:
    with st.spinner("Loading duties config from BigQuery..."):
        df_loaded = fetch_config_df(mapped_code)
        st.session_state.ddp_cfg_original = df_loaded.copy()
        st.session_state.ddp_cfg_edited = df_loaded.copy()

st.markdown("## Table Editor")
editor_df = st.session_state.ddp_cfg_edited.copy()
if "application_date" in editor_df.columns:
    editor_df["application_date"] = pd.to_datetime(editor_df["application_date"], errors="coerce")

edited = st.data_editor(
    editor_df,
    num_rows="dynamic",
    use_container_width=True,
    height=520,
    column_config={
        "duty_recalculation": st.column_config.NumberColumn("duty_recalculation", min_value=0, max_value=1, step=1),
        "ddp_fix": st.column_config.NumberColumn("ddp_fix", step=0.01, format="%.4f"),
        "ddp_perc": st.column_config.NumberColumn("ddp_perc", step=0.0001, format="%.6f"),
        "treshold_amount": st.column_config.NumberColumn("treshold_amount", step=0.01, format="%.4f"),
        "application_date": st.column_config.DateColumn("application_date", format="YYYY-MM-DD"),
    },
    key="ddp_cfg_editor",
)

edited_processed = edited.copy()
if "application_date" in edited_processed.columns:
    try:
        dt = pd.to_datetime(edited_processed["application_date"], errors="coerce")
        edited_processed["application_date"] = dt.dt.strftime("%Y-%m-%d").fillna("")
    except Exception:
        pass
st.session_state.ddp_cfg_edited = edited_processed

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    review_btn = st.button("Review Changes", type="secondary", use_container_width=True)
with c2:
    apply_btn = st.button("Apply Changes", type="primary", use_container_width=True)

if review_btn:
    canon = canonicalize_df(st.session_state.ddp_cfg_edited)
    errs = validate_df(canon)
    if errs:
        for e in errs:
            st.error(e)
    else:
        ins, upd, dele = diff_configs(st.session_state.ddp_cfg_original, canon)
        st.info(f"To insert: {len(ins)} | To update: {len(upd)} | To delete: {len(dele)}")
        if ins:
            st.markdown("#### Inserts (preview)")
            st.dataframe(pd.DataFrame([r.to_dict() for r in ins]).head(50), use_container_width=True)
        if upd:
            st.markdown("#### Updates (preview)")
            st.dataframe(pd.DataFrame([r.to_dict() for r in upd]).head(50), use_container_width=True)
        if dele:
            st.markdown("#### Deletes (preview)")
            st.dataframe(pd.DataFrame([r.to_dict() for r in dele]).head(50), use_container_width=True)

if apply_btn:
    with st.spinner("Validating and applying changes to BigQuery..."):
        canon = canonicalize_df(st.session_state.ddp_cfg_edited)
        errs = validate_df(canon)
        if errs:
            for e in errs:
                st.error(e)
        else:
            ins, upd, dele = diff_configs(st.session_state.ddp_cfg_original, canon)
            deleted = delete_rows(dele) if dele else 0
            upserted = upsert_rows(ins + upd) if (ins or upd) else 0
            st.success(f"Applied changes. Deleted: {deleted} | Upserted: {upserted}")
            # Refresh original/edited from DB
            refreshed = fetch_config_df(resolve_brand_code(st.session_state.get("brand")))
            st.session_state.ddp_cfg_original = refreshed.copy()
            st.session_state.ddp_cfg_edited = refreshed.copy()
            st.rerun()



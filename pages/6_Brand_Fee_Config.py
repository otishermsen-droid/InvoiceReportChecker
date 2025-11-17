import streamlit as st
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime


from common import client, get_supported_brands

TABLE_FQN = "tlg-business-intelligence-prd.config.brand_fee"

st.set_page_config(page_title="Brand Fee Config", layout="wide")
st.title("⚙️ Brand Fee Configuration")

# Style primary buttons (Load Table, Apply Changes) as green
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
/* Fallback for older Streamlit builds */
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


def normalize_country_list(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        # split by comma, uppercase, trim and sort for deterministic key
        parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
        parts = sorted(set(parts))
        return ",".join(parts)
    # Fallback: try list-like
    try:
        seq = list(value)
        parts = [str(p).strip().upper() for p in seq if str(p).strip()]
        parts = sorted(set(parts))
        return ",".join(parts)
    except Exception:
        return str(value).strip().upper()


# Base key columns (endless_aisle will be included only for CV)
BASE_KEY_COLUMNS = [
    "brand",
    "erp_entity",
    "location_code",
    "fee_calc_base",
    "threshold_base",
    "min_threshold_ytd",
    "max_threshold_ytd",
    "shipping_country_inclusion",
    "shipping_country_exclusion",
    "reset_date",
]

# Full key columns list (used for ALL_COLUMNS ordering)
KEY_COLUMNS = BASE_KEY_COLUMNS + ["endless_aisle"]

NON_KEY_COLUMNS = [
    "fee_perc",
]

# Include optional unique row identifier at the front for convenience if present in BQ
ALL_COLUMNS = ["row_key"] + KEY_COLUMNS + NON_KEY_COLUMNS


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ensure all expected columns exist
    for c in ALL_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    # Normalize row_key as trimmed string (do not change case to respect GUIDs)
    if "row_key" in out.columns:
        out["row_key"] = out["row_key"].astype(str).str.strip().where(out["row_key"].notna(), None)

    # Upper/lower casing and trimming for string keys (exclude array-typed columns)
    for c in ["brand", "erp_entity", "fee_calc_base", "threshold_base"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.upper().where(out[c].notna(), None)

    # Array-like columns normalized as canon comma-separated strings
    for c in ["location_code", "shipping_country_inclusion", "shipping_country_exclusion"]:
        if c in out.columns:
            out[c] = out[c].apply(normalize_country_list)

    # Thresholds to float (nullable)
    for c in ["min_threshold_ytd", "max_threshold_ytd"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # endless_aisle to nullable int {0,1}
    if "endless_aisle" in out.columns:
        ea = pd.to_numeric(out["endless_aisle"], errors="coerce").round().astype("Int64")
        # restrict to {0,1}
        out["endless_aisle"] = ea.where(ea.isin([0, 1]), pd.NA)

    # reset_date to YYYY-MM-DD (string) or empty
    if "reset_date" in out.columns:
        def _fmt_date(x):
            if pd.isna(x) or x is None or str(x).strip() == "":
                return ""
            try:
                return pd.to_datetime(x).strftime("%Y-%m-%d")
            except Exception:
                return ""
        out["reset_date"] = out["reset_date"].apply(_fmt_date)

    # fee_perc to float and clip 0..1
    if "fee_perc" in out.columns:
        fp = pd.to_numeric(out["fee_perc"], errors="coerce")
        out["fee_perc"] = fp

    # Keep only expected columns in a stable order
    out = out[ALL_COLUMNS]
    return out


def key_tuple(row: pd.Series) -> Tuple:
    # Prefer explicit row_key when provided
    rk = str(row.get("row_key") or "").strip()
    if rk:
        return (rk,)
    brand_key = str(row.get("brand") or "").strip().upper()
    cols = list(BASE_KEY_COLUMNS)
    if brand_key == "CV":
        cols = cols[:]
        cols.insert(len(cols) - 1, "endless_aisle")  # before reset_date
    return tuple(row.get(c) for c in cols)


def fetch_config_df(brand: Optional[str]) -> pd.DataFrame:
    where = []
    params = []
    if isinstance(brand, str) and brand.strip():
        where.append("brand = @brand")
        params.append(("brand", brand.strip().upper()))

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
    # fee_perc mandatory and 0..1
    if "fee_perc" in df.columns:
        bad = df["fee_perc"].isna() | (pd.to_numeric(df["fee_perc"], errors="coerce") < 0) | (pd.to_numeric(df["fee_perc"], errors="coerce") > 1)
        if bad.any():
            errors.append("fee_perc must be numeric in [0, 1] for all rows.")
    # thresholds logical
    mins = pd.to_numeric(df.get("min_threshold_ytd"), errors="coerce")
    maxs = pd.to_numeric(df.get("max_threshold_ytd"), errors="coerce")
    mask = mins.notna() & maxs.notna() & (mins > maxs)
    if mask.any():
        errors.append("min_threshold_ytd must be <= max_threshold_ytd when both are provided.")
    # composite key uniqueness (brand-aware: endless_aisle only for CV)
    keys = df.apply(key_tuple, axis=1)
    if keys.duplicated().any():
        errors.append("Composite key must be unique. Found duplicate key(s) in the edited data.")
    return errors


def diff_configs(original: pd.DataFrame, edited: pd.DataFrame):
    # Returns (to_insert, to_update, to_delete, key_changed_pairs)
    orig_canon = canonicalize_df(original)
    edit_canon = canonicalize_df(edited)

    orig_map: Dict[Tuple, pd.Series] = {key_tuple(r): r for _, r in orig_canon.iterrows()}
    edit_map: Dict[Tuple, pd.Series] = {key_tuple(r): r for _, r in edit_canon.iterrows()}

    orig_keys = set(orig_map.keys())
    edit_keys = set(edit_map.keys())

    to_delete_keys = sorted(list(orig_keys - edit_keys))
    to_insert_keys = sorted(list(edit_keys - orig_keys))
    common_keys = orig_keys & edit_keys

    to_update_keys = []
    for k in common_keys:
        o = orig_map[k]
        e = edit_map[k]
        # If matched by row_key (len==1), compare ALL columns except row_key.
        # Otherwise (composite match), compare only NON_KEY_COLUMNS to avoid mutating keys via UPDATE.
        changed = False
        if isinstance(k, tuple) and len(k) == 1:
            for c in [c for c in ALL_COLUMNS if c != "row_key"]:
                ov = o.get(c)
                ev = e.get(c)
                if (pd.isna(ov) and pd.isna(ev)) or (ov == ev):
                    continue
                changed = True
                break
        else:
            for c in NON_KEY_COLUMNS:
                if pd.isna(o.get(c)) and pd.isna(e.get(c)):
                    continue
                if (o.get(c) != e.get(c)):
                    changed = True
                    break
        if changed:
            to_update_keys.append(k)

    to_delete = [orig_map[k] for k in to_delete_keys]
    to_insert = [edit_map[k] for k in to_insert_keys]
    to_update = [edit_map[k] for k in to_update_keys]

    return to_insert, to_update, to_delete


def delete_rows(rows: List[pd.Series]) -> int:
    from google.cloud import bigquery
    total = 0
    for r in rows:
        # Prepare array params for country lists
        def _as_array(val: Any) -> Optional[List[str]]:
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                return None
            parts = [p.strip().upper() for p in str(val).split(",") if p.strip()]
            return parts or None

        is_cv = (str(r.get("brand") or "").strip().upper() == "CV")
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
    ((brand IS NULL AND @brand IS NULL) OR UPPER(brand) = UPPER(@brand))
    AND ((erp_entity IS NULL AND @erp_entity IS NULL) OR UPPER(erp_entity) = UPPER(@erp_entity))
    AND (
         (location_code IS NULL AND @location_code IS NULL)
      OR (
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(location_code) x)
           IS NOT DISTINCT FROM
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(@location_code) x)
         )
    )
    AND ((fee_calc_base IS NULL AND @fee_calc_base IS NULL) OR UPPER(fee_calc_base) = UPPER(@fee_calc_base))
    AND ((threshold_base IS NULL AND @threshold_base IS NULL) OR UPPER(threshold_base) = UPPER(@threshold_base))
    AND (SAFE_CAST(min_threshold_ytd AS NUMERIC) IS NOT DISTINCT FROM @min_threshold_ytd)
    AND (SAFE_CAST(max_threshold_ytd AS NUMERIC) IS NOT DISTINCT FROM @max_threshold_ytd)
    AND (
         (shipping_country_inclusion IS NULL AND @shipping_country_inclusion IS NULL)
      OR (
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(shipping_country_inclusion) x)
           IS NOT DISTINCT FROM
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(@shipping_country_inclusion) x)
         )
    )
    AND (
         (shipping_country_exclusion IS NULL AND @shipping_country_exclusion IS NULL)
      OR (
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(shipping_country_exclusion) x)
           IS NOT DISTINCT FROM
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(@shipping_country_exclusion) x)
         )
    )
    AND ( @is_cv = FALSE OR (SAFE_CAST(endless_aisle AS INT64) IS NOT DISTINCT FROM @endless_aisle) )
    AND (reset_date IS NOT DISTINCT FROM @reset_date)
  )
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("has_row_key", "BOOL", row_key is not None),
                bigquery.ScalarQueryParameter("row_key", "STRING", row_key),
                bigquery.ScalarQueryParameter("brand", "STRING", r.get("brand")),
                bigquery.ScalarQueryParameter("erp_entity", "STRING", r.get("erp_entity")),
                bigquery.ArrayQueryParameter("location_code", "STRING", _as_array(r.get("location_code"))),
                bigquery.ScalarQueryParameter("fee_calc_base", "STRING", r.get("fee_calc_base")),
                bigquery.ScalarQueryParameter("threshold_base", "STRING", r.get("threshold_base")),
                bigquery.ScalarQueryParameter("min_threshold_ytd", "NUMERIC", None if pd.isna(r.get("min_threshold_ytd")) else float(r.get("min_threshold_ytd"))),
                bigquery.ScalarQueryParameter("max_threshold_ytd", "NUMERIC", None if pd.isna(r.get("max_threshold_ytd")) else float(r.get("max_threshold_ytd"))),
                bigquery.ArrayQueryParameter("shipping_country_inclusion", "STRING", _as_array(r.get("shipping_country_inclusion"))),
                bigquery.ArrayQueryParameter("shipping_country_exclusion", "STRING", _as_array(r.get("shipping_country_exclusion"))),
                bigquery.ScalarQueryParameter("endless_aisle", "INT64", None if pd.isna(r.get("endless_aisle")) else int(r.get("endless_aisle"))),
                bigquery.ScalarQueryParameter("is_cv", "BOOL", is_cv),
                bigquery.ScalarQueryParameter("reset_date", "DATETIME", None if (r.get("reset_date") in (None, "")) else r.get("reset_date")),
            ]
        )
        res = client.query(sql, job_config=job_config).result()
        total += res.total_rows or 0
    return total


def upsert_rows(rows: List[pd.Series]) -> int:
    from google.cloud import bigquery
    total = 0
    for r in rows:
        # Prepare array params for country lists
        def _as_array(val: Any) -> Optional[List[str]]:
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                return None
            parts = [p.strip().upper() for p in str(val).split(",") if p.strip()]
            return parts or None

        row_key = (str(r.get("row_key") or "").strip() or None)

        sql = f"""
MERGE `{TABLE_FQN}` T
USING (SELECT
    @row_key AS row_key,
    @brand AS brand,
    @erp_entity AS erp_entity,
    @location_code AS location_code,
    @fee_calc_base AS fee_calc_base,
    @threshold_base AS threshold_base,
    @min_threshold_ytd AS min_threshold_ytd,
    @max_threshold_ytd AS max_threshold_ytd,
    @shipping_country_inclusion AS shipping_country_inclusion,
    @shipping_country_exclusion AS shipping_country_exclusion,
    @endless_aisle AS endless_aisle,
    @reset_date AS reset_date,
    @fee_perc AS fee_perc
) S
ON (
  (S.row_key IS NOT NULL AND T.row_key = S.row_key)
  OR
  (
    S.row_key IS NULL AND
    ((T.brand IS NULL AND S.brand IS NULL) OR UPPER(T.brand) = UPPER(S.brand)) AND
    ((T.erp_entity IS NULL AND S.erp_entity IS NULL) OR UPPER(T.erp_entity) = UPPER(S.erp_entity)) AND
    (
         (T.location_code IS NULL AND S.location_code IS NULL)
      OR (
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(T.location_code) x)
           IS NOT DISTINCT FROM
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(S.location_code) x)
         )
    ) AND
    ((T.fee_calc_base IS NULL AND S.fee_calc_base IS NULL) OR UPPER(T.fee_calc_base) = UPPER(S.fee_calc_base)) AND
    ((T.threshold_base IS NULL AND S.threshold_base IS NULL) OR UPPER(T.threshold_base) = UPPER(S.threshold_base)) AND
    (SAFE_CAST(T.min_threshold_ytd AS NUMERIC) IS NOT DISTINCT FROM S.min_threshold_ytd) AND
    (SAFE_CAST(T.max_threshold_ytd AS NUMERIC) IS NOT DISTINCT FROM S.max_threshold_ytd) AND
    (
         (T.shipping_country_inclusion IS NULL AND S.shipping_country_inclusion IS NULL)
      OR (
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(T.shipping_country_inclusion) x)
           IS NOT DISTINCT FROM
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(S.shipping_country_inclusion) x)
         )
    ) AND
    (
         (T.shipping_country_exclusion IS NULL AND S.shipping_country_exclusion IS NULL)
      OR (
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(T.shipping_country_exclusion) x)
           IS NOT DISTINCT FROM
           (SELECT STRING_AGG(UPPER(TRIM(x)), ',' ORDER BY UPPER(TRIM(x))) FROM UNNEST(S.shipping_country_exclusion) x)
         )
    ) AND
    ( (UPPER(S.brand) != 'CV') OR (SAFE_CAST(T.endless_aisle AS INT64) IS NOT DISTINCT FROM SAFE_CAST(S.endless_aisle AS INT64)) ) AND
    (T.reset_date IS NOT DISTINCT FROM S.reset_date)
  )
)
WHEN MATCHED THEN UPDATE SET
  -- If matching by row_key, allow updating all fields; otherwise keep keys unchanged
  brand = CASE WHEN S.row_key IS NOT NULL THEN S.brand ELSE T.brand END,
  erp_entity = CASE WHEN S.row_key IS NOT NULL THEN S.erp_entity ELSE T.erp_entity END,
  location_code = CASE WHEN S.row_key IS NOT NULL THEN S.location_code ELSE T.location_code END,
  fee_calc_base = CASE WHEN S.row_key IS NOT NULL THEN S.fee_calc_base ELSE T.fee_calc_base END,
  threshold_base = CASE WHEN S.row_key IS NOT NULL THEN S.threshold_base ELSE T.threshold_base END,
  min_threshold_ytd = CASE WHEN S.row_key IS NOT NULL THEN S.min_threshold_ytd ELSE T.min_threshold_ytd END,
  max_threshold_ytd = CASE WHEN S.row_key IS NOT NULL THEN S.max_threshold_ytd ELSE T.max_threshold_ytd END,
  shipping_country_inclusion = CASE WHEN S.row_key IS NOT NULL THEN S.shipping_country_inclusion ELSE T.shipping_country_inclusion END,
  shipping_country_exclusion = CASE WHEN S.row_key IS NOT NULL THEN S.shipping_country_exclusion ELSE T.shipping_country_exclusion END,
  endless_aisle = CASE WHEN S.row_key IS NOT NULL THEN S.endless_aisle ELSE T.endless_aisle END,
  reset_date = CASE WHEN S.row_key IS NOT NULL THEN S.reset_date ELSE T.reset_date END,
  fee_perc = S.fee_perc
WHEN NOT MATCHED THEN INSERT
  (row_key, brand, erp_entity, location_code, fee_calc_base, threshold_base, min_threshold_ytd, max_threshold_ytd, shipping_country_inclusion, shipping_country_exclusion, endless_aisle, reset_date, fee_perc)
  VALUES
  (S.row_key, S.brand, S.erp_entity, S.location_code, S.fee_calc_base, S.threshold_base, S.min_threshold_ytd, S.max_threshold_ytd, S.shipping_country_inclusion, S.shipping_country_exclusion, S.endless_aisle, S.reset_date, S.fee_perc)
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("row_key", "STRING", row_key),
                bigquery.ScalarQueryParameter("brand", "STRING", r.get("brand")),
                bigquery.ScalarQueryParameter("erp_entity", "STRING", r.get("erp_entity")),
                bigquery.ArrayQueryParameter("location_code", "STRING", _as_array(r.get("location_code"))),
                bigquery.ScalarQueryParameter("fee_calc_base", "STRING", r.get("fee_calc_base")),
                bigquery.ScalarQueryParameter("threshold_base", "STRING", r.get("threshold_base")),
                bigquery.ScalarQueryParameter("min_threshold_ytd", "NUMERIC", None if pd.isna(r.get("min_threshold_ytd")) else float(r.get("min_threshold_ytd"))),
                bigquery.ScalarQueryParameter("max_threshold_ytd", "NUMERIC", None if pd.isna(r.get("max_threshold_ytd")) else float(r.get("max_threshold_ytd"))),
                bigquery.ArrayQueryParameter("shipping_country_inclusion", "STRING", _as_array(r.get("shipping_country_inclusion"))),
                bigquery.ArrayQueryParameter("shipping_country_exclusion", "STRING", _as_array(r.get("shipping_country_exclusion"))),
                bigquery.ScalarQueryParameter("endless_aisle", "INT64", None if pd.isna(r.get("endless_aisle")) else int(r.get("endless_aisle"))),
                bigquery.ScalarQueryParameter("reset_date", "DATETIME", None if (r.get("reset_date") in (None, "")) else r.get("reset_date")),
                bigquery.ScalarQueryParameter("fee_perc", "NUMERIC", None if pd.isna(r.get("fee_perc")) else float(r.get("fee_perc"))),
            ]
        )
        client.query(sql, job_config=job_config).result()
        total += 1
    return total


with st.sidebar:
    st.markdown("### Filter")
    brand_options = get_supported_brands()
    # Try to preselect current brand
    default_idx = 0
    pre = st.session_state.get("brand")
    if isinstance(pre, str) and pre in brand_options:
        default_idx = brand_options.index(pre)
    selected_brand = st.selectbox("Brand", options=brand_options, index=default_idx)
    st.session_state["brand"] = selected_brand
    load_btn = st.button("Load Table", type="primary", use_container_width=True)

    # Bottom configuration section (removed per request)

if "brand_fee_original" not in st.session_state:
    st.session_state.brand_fee_original = pd.DataFrame(columns=ALL_COLUMNS)
if "brand_fee_edited" not in st.session_state:
    st.session_state.brand_fee_edited = pd.DataFrame(columns=ALL_COLUMNS)

if load_btn:
    with st.spinner("Loading brand_fee from BigQuery..."):
        df_loaded = fetch_config_df(selected_brand)
        st.session_state.brand_fee_original = df_loaded.copy()
        st.session_state.brand_fee_edited = df_loaded.copy()

st.markdown("## Table Editor")
# Convert reset_date to datetime for editor compatibility
editor_df = st.session_state.brand_fee_edited.copy()
if "reset_date" in editor_df.columns:
    editor_df["reset_date"] = pd.to_datetime(editor_df["reset_date"], errors="coerce")

edited = st.data_editor(
    editor_df,
    num_rows="dynamic",
    use_container_width=True,
    height=480,
    column_config={
        "fee_perc": st.column_config.NumberColumn("fee_perc", min_value=0.0, max_value=1.0, step=0.01, format="%.4f"),
        "min_threshold_ytd": st.column_config.NumberColumn("min_threshold_ytd", step=0.01, format="%.2f"),
        "max_threshold_ytd": st.column_config.NumberColumn("max_threshold_ytd", step=0.01, format="%.2f"),
        "endless_aisle": st.column_config.NumberColumn("endless_aisle", min_value=0, max_value=1, step=1),
        "reset_date": st.column_config.DateColumn("reset_date", format="YYYY-MM-DD"),
    },
    key="brand_fee_editor",
)
# Convert reset_date back to canonical string format for storage
edited_processed = edited.copy()
if "reset_date" in edited_processed.columns:
    try:
        dt = pd.to_datetime(edited_processed["reset_date"], errors="coerce")
        edited_processed["reset_date"] = dt.dt.strftime("%Y-%m-%d").fillna("")
    except Exception:
        pass
st.session_state.brand_fee_edited = edited_processed


c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    review_btn = st.button("Review Changes", type="secondary", use_container_width=True)
with c2:
    apply_btn = st.button("Apply Changes", type="primary", use_container_width=True)

if review_btn:
    canon = canonicalize_df(st.session_state.brand_fee_edited)
    errs = validate_df(canon)
    if errs:
        for e in errs:
            st.error(e)
    else:
        ins, upd, dele = diff_configs(st.session_state.brand_fee_original, canon)
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
        canon = canonicalize_df(st.session_state.brand_fee_edited)
        errs = validate_df(canon)
        if errs:
            for e in errs:
                st.error(e)
        else:
            ins, upd, dele = diff_configs(st.session_state.brand_fee_original, canon)
            deleted = delete_rows(dele) if dele else 0
            upserted = upsert_rows(ins + upd) if (ins or upd) else 0
            st.success(f"Applied changes. Deleted: {deleted} | Upserted: {upserted}")
            # Refresh original/edited from DB
            refreshed = fetch_config_df(st.session_state.get("brand"))
            st.session_state.brand_fee_original = refreshed.copy()
            st.session_state.brand_fee_edited = refreshed.copy()
            st.rerun()



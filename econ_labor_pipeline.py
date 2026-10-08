"""Person 3 pipeline: FRED + BLS economic and labor data -> county x quarter tables.

Usage (from the repository root):
    python src/econ_labor_pipeline.py            # download raw files, then build processed tables
    python src/econ_labor_pipeline.py download   # only download raw snapshots
    python src/econ_labor_pipeline.py build      # only rebuild processed tables from saved raw files

Optional API keys (never commit them):
    BLS_API_KEY   - recommended. Without it the script uses the BLS v1 API (smaller batches).
    FRED needs no key: series are downloaded from the public fredgraph CSV endpoint.

Outputs
    data/raw/FRED/<retrieved_date>/<SERIES>.csv          unchanged FRED downloads
    data/raw/BLS/<retrieved_date>/bls_response_<n>.json  unchanged BLS API responses
    data/raw/raw_manifest_econ_labor.csv                 URL, series, retrieval time, SHA-256 per raw file
    data/processed/FRED/national_quarterly.csv           national variables, one row per quarter
    data/processed/BLS/state_labor_quarterly.csv         Ohio (state-level) labor variables per quarter
    data/processed/BLS/county_labor_quarterly.csv        county labor variables per county x quarter
    data/processed/econ_labor_county_quarter.csv         combined table: 9 counties x 44 quarters
    data/processed/econ_labor_variable_definitions.csv   variable definition table
    data/processed/econ_labor_coverage.csv               missing cells per variable
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FRED = ROOT / "data" / "raw" / "FRED"
RAW_BLS = ROOT / "data" / "raw" / "BLS"
MANIFEST = ROOT / "data" / "raw" / "raw_manifest_econ_labor.csv"
OUT = ROOT / "data" / "processed"

# Study window. Raw downloads start one year earlier so 2015 year-over-year inflation can be computed.
START_YEAR, END_YEAR = 2015, 2025
RAW_START_YEAR = START_YEAR - 1

COUNTIES = {
    "39017": "Butler",
    "39035": "Cuyahoga",
    "39049": "Franklin",
    "39061": "Hamilton",
    "39095": "Lucas",
    "39113": "Montgomery",
    "39151": "Stark",
    "39153": "Summit",
    "39165": "Warren",
}

# Team decision (2026-10-08): monthly series accept a quarter with 2 of 3 months, flagged as partial.
# This covers 2025Q4, when October 2025 data were not published because of the federal shutdown.
MIN_MONTHS = 2

# FRED series: id -> (output column, native frequency, minimum obs to accept, obs in a full quarter)
FRED_SERIES = {
    "MORTGAGE30US": ("nat_mortgage_rate_30y_pct", "weekly", 12, 12),
    "CPIAUCSL": ("nat_cpi_index", "monthly", MIN_MONTHS, 3),
    "DGS3": ("nat_treasury_3y_pct", "daily", 55, 55),  # team decision: low-risk return benchmark
    "DGS10": ("nat_treasury_10y_pct", "daily", 55, 55),
    "TB3MS": ("nat_tbill_3m_pct", "monthly", MIN_MONTHS, 3),
}

# BLS LAUS measure codes
LAUS_MEASURES = {"04": "unemployed", "05": "employed", "06": "labor_force"}
STATE_PREFIX = "LASST39" + "0" * 11  # Ohio, seasonally adjusted, e.g. LASST390000000000003
COUNTY_PREFIX = "LAUCN{fips}00000000"  # county, not seasonally adjusted


def bls_series_ids() -> dict[str, tuple[str, str, str]]:
    """Return {series_id: (geo_level, fips, measure)}."""
    ids = {}
    for code, measure in LAUS_MEASURES.items():
        ids[f"{STATE_PREFIX}{code}"] = ("state", "39", measure)
        for fips in COUNTIES:
            ids[f"{COUNTY_PREFIX.format(fips=fips)}{code}"] = ("county", fips, measure)
    bad = [sid for sid in ids if len(sid) != 20]
    assert not bad, f"LAUS series IDs must be 20 characters: {bad}"
    return ids


# ----------------------------------------------------------------------------- download

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_manifest(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if MANIFEST.exists():
        old = pd.read_csv(MANIFEST, dtype=str)
        old = old[~old["raw_file"].isin(df["raw_file"])]  # same-day re-download replaces its own rows
        df = pd.concat([old, df], ignore_index=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)


def download() -> None:
    import requests

    today = dt.date.today().isoformat()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    manifest = []

    fred_dir = RAW_FRED / today
    fred_dir.mkdir(parents=True, exist_ok=True)
    for series in FRED_SERIES:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series}&cosd={RAW_START_YEAR}-01-01&coed={END_YEAR}-12-31"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        path = fred_dir / f"{series}.csv"
        path.write_bytes(resp.content)
        manifest.append(dict(source="FRED", series_id=series, raw_file=str(path.relative_to(ROOT)),
                             url=url, retrieved_at_utc=now, sha256=_sha256(path)))
        print(f"FRED {series}: saved {path.relative_to(ROOT)}")

    key = os.environ.get("BLS_API_KEY", "").strip()
    if key:
        endpoint, max_series, max_years = "https://api.bls.gov/publicAPI/v2/timeseries/data/", 50, 20
    else:
        print("No BLS_API_KEY set: using BLS v1 API (25 series and 10 years per request).")
        endpoint, max_series, max_years = "https://api.bls.gov/publicAPI/v1/timeseries/data/", 25, 10

    ids = list(bls_series_ids())
    year_windows = [(y, min(y + max_years - 1, END_YEAR)) for y in range(RAW_START_YEAR, END_YEAR + 1, max_years)]
    bls_dir = RAW_BLS / today
    bls_dir.mkdir(parents=True, exist_ok=True)
    for old in bls_dir.glob("bls_response_*.json"):  # same-day re-download replaces that day's snapshot
        old.unlink()
    n = 0
    for start, end in year_windows:
        for i in range(0, len(ids), max_series):
            batch = ids[i:i + max_series]
            payload = {"seriesid": batch, "startyear": str(start), "endyear": str(end)}
            if key:
                payload["registrationkey"] = key
            resp = requests.post(endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS request failed: {body.get('status')} {body.get('message')}")
            for msg in body.get("message", []):
                print(f"BLS message: {msg}")
            empty = [s["seriesID"] for s in body["Results"]["series"] if not s.get("data")]
            if empty:
                print(f"WARNING: BLS returned no data for {len(empty)} series: {empty}")
            n += 1
            path = bls_dir / f"bls_response_{n:02d}.json"
            path.write_text(json.dumps(body, indent=1))
            manifest.append(dict(source="BLS LAUS", series_id=";".join(batch), raw_file=str(path.relative_to(ROOT)),
                                 url=endpoint, retrieved_at_utc=now, sha256=_sha256(path)))
            print(f"BLS batch {n}: {len(batch)} series, {start}-{end}")
            time.sleep(1)

    _append_manifest(manifest)
    print(f"Manifest updated: {MANIFEST.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- load raw

def _latest_dir(base: Path) -> Path:
    dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if not dirs:
        raise FileNotFoundError(f"No raw snapshot folders in {base}. Run the download step first.")
    return dirs[-1]


def load_fred(folder: Path | None = None) -> pd.DataFrame:
    """Long table: date, series, value (NaN where FRED reports no value)."""
    folder = folder or _latest_dir(RAW_FRED)
    frames = []
    for series in FRED_SERIES:
        raw = pd.read_csv(folder / f"{series}.csv", dtype=str)
        date_col, value_col = raw.columns[0], raw.columns[1]
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(raw[date_col]),
            "series": series,
            "value": pd.to_numeric(raw[value_col].replace({".": None, "": None}), errors="coerce"),
        }))
    return pd.concat(frames, ignore_index=True)


def load_bls(folder: Path | None = None) -> pd.DataFrame:
    """Long table: series, geo_level, fips, measure, year, month, value, preliminary."""
    folder = folder or _latest_dir(RAW_BLS)
    meta = bls_series_ids()
    rows = []
    for path in sorted(folder.glob("bls_response_*.json")):
        body = json.loads(path.read_text())
        for s in body["Results"]["series"]:
            sid = s["seriesID"]
            geo_level, fips, measure = meta[sid]
            for obs in s["data"]:
                period = obs["period"]
                if not period.startswith("M") or period == "M13":  # M13 = annual average
                    continue
                codes = {f.get("code") for f in obs.get("footnotes", []) if f}
                rows.append(dict(series=sid, geo_level=geo_level, fips=fips, measure=measure,
                                 year=int(obs["year"]), month=int(period[1:]),
                                 value=pd.to_numeric(obs["value"].replace(",", ""), errors="coerce"),
                                 preliminary="P" in codes))
    df = pd.DataFrame(rows)
    dupes = df.duplicated(["series", "year", "month"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate BLS series-month observations across raw files")
    return df


# ----------------------------------------------------------------------------- build

def quarter_skeleton() -> pd.DataFrame:
    q = pd.period_range(f"{START_YEAR}Q1", f"{END_YEAR}Q4", freq="Q")
    return pd.DataFrame({"year": q.year, "quarter": q.quarter})


def build_national(fred: pd.DataFrame) -> pd.DataFrame:
    fred = fred.dropna(subset=["value"]).copy()
    fred["year"], fred["quarter"] = fred["date"].dt.year, fred["date"].dt.quarter
    agg = fred.groupby(["series", "year", "quarter"])["value"].agg(["mean", "count"]).reset_index()
    q = pd.period_range(f"{RAW_START_YEAR}Q1", f"{END_YEAR}Q4", freq="Q")
    out = pd.DataFrame({"year": q.year, "quarter": q.quarter})
    for series, (col, _, min_obs, full_obs) in FRED_SERIES.items():
        s = agg[agg["series"] == series][["year", "quarter", "mean", "count"]]
        out = out.merge(s, on=["year", "quarter"], how="left")
        out["count"] = out["count"].fillna(0).astype(int)
        out[col] = out["mean"].where(out["count"] >= min_obs)  # too few observations -> missing
        out[f"{col}_n_obs"] = out["count"]
        out[f"{col}_partial"] = (out["count"] >= min_obs) & (out["count"] < full_obs)
        out = out.drop(columns=["mean", "count"])
    # Year-over-year CPI inflation: same quarter one year earlier, consecutive rows only.
    out = out.sort_values(["year", "quarter"]).reset_index(drop=True)
    lag = out["nat_cpi_index"].shift(4)
    lag_ok = (out["year"].shift(4) == out["year"] - 1) & (out["quarter"].shift(4) == out["quarter"])
    out["nat_inflation_yoy_pct"] = (100 * (out["nat_cpi_index"] / lag - 1)).where(lag_ok)
    lag_partial = out["nat_cpi_index_partial"].shift(4).fillna(False).astype(bool)
    out["nat_inflation_yoy_pct_partial"] = (out["nat_cpi_index_partial"] | lag_partial) & out["nat_inflation_yoy_pct"].notna()
    return out[out["year"] >= START_YEAR].reset_index(drop=True)


def build_labor(bls: pd.DataFrame, geo_level: str) -> pd.DataFrame:
    df = bls[(bls["geo_level"] == geo_level) & bls["year"].between(START_YEAR, END_YEAR)].copy()
    if df.empty:
        raise ValueError(f"No BLS {geo_level}-level observations in the raw files. "
                         "Check the BLS messages printed during download, then re-run the download step.")
    df["quarter"] = (df["month"] - 1) // 3 + 1
    wide = df.pivot_table(index=["fips", "year", "quarter", "month"], columns="measure",
                          values="value", aggfunc="first").reset_index()
    prelim = df.groupby(["fips", "year", "quarter"])["preliminary"].any().rename("any_preliminary")
    measures = list(LAUS_MEASURES.values())
    complete_month = wide[measures].notna().all(axis=1)
    wide = wide[complete_month]
    agg = wide.groupby(["fips", "year", "quarter"]).agg(
        **{m: (m, "mean") for m in measures}, n_months=("month", "nunique")).reset_index()
    fips_list = ["39"] if geo_level == "state" else list(COUNTIES)
    skel = quarter_skeleton().merge(pd.DataFrame({"fips": fips_list}), how="cross")
    out = skel.merge(agg, on=["fips", "year", "quarter"], how="left").merge(
        prelim.reset_index(), on=["fips", "year", "quarter"], how="left")
    out["n_months"] = out["n_months"].fillna(0).astype(int)
    out["any_preliminary"] = out["any_preliminary"].fillna(False).astype(bool)
    accepted = out["n_months"] >= MIN_MONTHS
    for m in measures:
        out[m] = out[m].where(accepted)  # fewer than MIN_MONTHS months -> missing
    # Rate from summed counts, never an average of monthly rates.
    out["unemployment_rate_pct"] = 100 * out["unemployed"] / out["labor_force"]
    out["partial_quarter"] = accepted & (out["n_months"] < 3)
    out["missing_reason"] = ""
    out.loc[~accepted & (out["n_months"] > 0), "missing_reason"] = "incomplete_period"
    out.loc[out["n_months"] == 0, "missing_reason"] = "not_available"
    prefix = "state_" if geo_level == "state" else "cty_"
    rename = {m: f"{prefix}{m}" for m in measures + ["unemployment_rate_pct", "n_months", "partial_quarter",
                                                      "any_preliminary", "missing_reason"]}
    out = out.rename(columns=rename)
    if geo_level == "county":
        out = out.rename(columns={"fips": "county_fips"})
        out["county_name"] = out["county_fips"].map(COUNTIES)
        keys = ["county_fips", "county_name", "year", "quarter"]
        out = out[keys + [c for c in out.columns if c not in keys]]
        out = out.sort_values(["county_fips", "year", "quarter"]).reset_index(drop=True)
    else:
        out = out.drop(columns="fips")
    return out


def variable_definitions() -> pd.DataFrame:
    fred_url = "https://fred.stlouisfed.org/series/"
    rows = [
        ("county_fips", "key", "5-digit county FIPS code (state 39 = Ohio)", "code", "", "", ""),
        ("county_name", "key", "County name without 'County'", "text", "", "", ""),
        ("year", "key", "Calendar year", "year", "", "", ""),
        ("quarter", "key", "Calendar quarter (1-4)", "quarter", "", "", ""),
        ("nat_mortgage_rate_30y_pct", "national", "30-year fixed mortgage average rate (Freddie Mac PMMS)", "percent",
         "FRED", "MORTGAGE30US", "Mean of weekly observations dated in the quarter; missing if fewer than 12"),
        ("nat_cpi_index", "national", "CPI for All Urban Consumers, all items, seasonally adjusted", "index 1982-84=100",
         "FRED", "CPIAUCSL", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("nat_inflation_yoy_pct", "national", "Year-over-year CPI inflation", "percent",
         "Derived", "CPIAUCSL", "100 x (CPI_q / CPI_same_quarter_prior_year - 1)"),
        ("nat_treasury_3y_pct", "national", "3-year Treasury constant-maturity yield; the team's low-risk return benchmark", "percent",
         "FRED", "DGS3", "Mean of daily values in the quarter; missing if fewer than 55 trading days"),
        ("nat_treasury_10y_pct", "national", "10-year Treasury constant-maturity yield", "percent",
         "FRED", "DGS10", "Mean of daily values in the quarter; missing if fewer than 55 trading days"),
        ("nat_tbill_3m_pct", "national", "3-month Treasury bill secondary market rate", "percent",
         "FRED", "TB3MS", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("state_unemployed", "state", "Ohio unemployed persons, seasonally adjusted", "persons",
         "BLS LAUS", "LASST390000000000004", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("state_employed", "state", "Ohio employed persons, seasonally adjusted", "persons",
         "BLS LAUS", "LASST390000000000005", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("state_labor_force", "state", "Ohio civilian labor force, seasonally adjusted", "persons",
         "BLS LAUS", "LASST390000000000006", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("state_unemployment_rate_pct", "state", "Ohio unemployment rate", "percent",
         "Derived", "LASST39...04 / 06", "100 x state_unemployed / state_labor_force"),
        ("cty_unemployed", "county", "Unemployed residents, not seasonally adjusted", "persons",
         "BLS LAUS", "LAUCN<fips>0000000004", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("cty_employed", "county", "Employed residents (household concept, not payroll jobs), not seasonally adjusted",
         "persons", "BLS LAUS", "LAUCN<fips>0000000005", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("cty_labor_force", "county", "Civilian labor force, not seasonally adjusted", "persons",
         "BLS LAUS", "LAUCN<fips>0000000006", "Mean of available monthly values; needs at least 2 of 3 months"),
        ("cty_unemployment_rate_pct", "county", "Unemployment rate", "percent",
         "Derived", "LAUCN<fips>...04 / 06", "100 x cty_unemployed / cty_labor_force (not an average of monthly rates)"),
        ("*_n_obs, *_n_months", "quality", "Number of source observations used in the quarterly value", "count", "", "", ""),
        ("*_partial, *_partial_quarter", "quality",
         "True if the quarterly value uses only 2 of 3 months (2025Q4: October 2025 not published)", "boolean", "", "", ""),
        ("*_any_preliminary", "quality", "True if any month in the quarter is flagged preliminary (P) by BLS", "boolean", "BLS", "", ""),
        ("*_missing_reason", "quality", "Why a labor value is missing: incomplete_period or not_available", "text", "", "", ""),
    ]
    df = pd.DataFrame(rows, columns=["variable", "geo_level", "description", "unit", "source", "series_id",
                                     "quarterly_rule"])
    df["source_url"] = df.apply(lambda r: fred_url + r.series_id if r.source == "FRED" else
                                ("https://www.bls.gov/lau/" if "BLS" in r.source else ""), axis=1)
    return df


def build() -> None:
    fred_dir, bls_dir = _latest_dir(RAW_FRED), _latest_dir(RAW_BLS)
    print(f"Building from raw snapshots: {fred_dir.relative_to(ROOT)}, {bls_dir.relative_to(ROOT)}")
    national = build_national(load_fred(fred_dir))
    bls = load_bls(bls_dir)
    state = build_labor(bls, "state")
    county = build_labor(bls, "county")

    combined = county.merge(national, on=["year", "quarter"], how="left", validate="many_to_one")
    combined = combined.merge(state, on=["year", "quarter"], how="left", validate="many_to_one")
    keys = ["county_fips", "year", "quarter"]
    assert not combined.duplicated(keys).any(), "Duplicate county x quarter rows"
    expected = len(COUNTIES) * (END_YEAR - START_YEAR + 1) * 4
    assert len(combined) == expected, f"Expected {expected} rows, got {len(combined)}"

    for sub in ("FRED", "BLS"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    national.to_csv(OUT / "FRED" / "national_quarterly.csv", index=False)
    state.to_csv(OUT / "BLS" / "state_labor_quarterly.csv", index=False)
    county.to_csv(OUT / "BLS" / "county_labor_quarterly.csv", index=False)
    combined.to_csv(OUT / "econ_labor_county_quarter.csv", index=False)
    variable_definitions().to_csv(OUT / "econ_labor_variable_definitions.csv", index=False)

    value_cols = [c for c in combined.columns if c not in keys + ["county_name"]
                  and not c.endswith(("_n_obs", "_n_months", "_any_preliminary", "_missing_reason",
                                      "_partial", "_partial_quarter"))]
    def partial_col(c):
        if c.startswith("cty_"):
            return "cty_partial_quarter"
        if c.startswith("state_"):
            return "state_partial_quarter"
        return f"{c}_partial" if f"{c}_partial" in combined else None

    cov = pd.DataFrame({
        "variable": value_cols,
        "missing_cells": [int(combined[c].isna().sum()) for c in value_cols],
        "partial_quarter_cells": [int(combined[partial_col(c)].sum()) if partial_col(c) else 0 for c in value_cols],
        "total_cells": len(combined),
        "missing_quarters": [", ".join(sorted({f"{y}Q{q}" for y, q in
                              combined.loc[combined[c].isna(), ["year", "quarter"]].itertuples(index=False)}))
                             for c in value_cols],
    })
    cov.to_csv(OUT / "econ_labor_coverage.csv", index=False)
    print(f"Combined table: {len(combined)} rows ({len(COUNTIES)} counties x {expected // len(COUNTIES)} quarters)")
    print(cov[["variable", "missing_cells", "partial_quarter_cells", "missing_quarters"]].to_string(index=False))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("all", "download"):
        download()
    if step in ("all", "build"):
        build()

"""Create an interpretable, leakage-safe county-quarter feature panel.

Run after the source integration script:
    python src/build_master_county_quarter.py
    python src/build_master_county_quarter_features.py

The output keeps source levels and creates only features known at the current
quarter. Columns ending in ``_ahead_`` are labels and appear last.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT SETTINGS AND INPUT/OUTPUT FILES
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"

SOURCE_PANEL_FILE = PROCESSED / "master_county_quarter.csv"
SOURCE_DICTIONARY_FILE = METADATA / "master_data_dictionary.csv"
FEATURE_PANEL_FILE = PROCESSED / "master_county_quarter_features.csv"
FEATURE_MISSINGNESS_FILE = PROCESSED / "master_county_quarter_features_missingness.csv"
FEATURE_DICTIONARY_FILE = METADATA / "master_feature_dictionary.csv"
QUALITY_SUMMARY_FILE = PROCESSED / "master_county_quarter_features_quality_summary.csv"

KEY_COLUMNS = ["county_fips", "county_name", "year", "quarter", "quarter_end_date"]
EXPECTED_FIPS = {"39017", "39035", "39049", "39061", "39095", "39113", "39151", "39153", "39165"}


# =============================================================================
# 2. READ AND VALIDATE THE CLEANED SOURCE PANEL
# =============================================================================

panel = pd.read_csv(SOURCE_PANEL_FILE, dtype={"county_fips": str})
panel["county_fips"] = panel["county_fips"].str.zfill(5)
panel = panel.sort_values(["county_fips", "year", "quarter"]).reset_index(drop=True)

assert len(panel) == 396, "Expected 9 counties x 44 quarters = 396 rows."
assert set(panel["county_fips"]) == EXPECTED_FIPS, "Unexpected county FIPS in source panel."
assert not panel.duplicated(["county_fips", "year", "quarter"]).any(), "Duplicate county-quarter keys found."
assert panel.groupby("county_fips").size().eq(44).all(), "Each county must have 44 quarters."
assert panel.groupby("county_fips")["quarter_end_date"].min().eq("2015-03-31").all()
assert panel.groupby("county_fips")["quarter_end_date"].max().eq("2025-12-31").all()


# =============================================================================
# 3. CREATE BACKWARD-LOOKING HOUSING MOMENTUM FEATURES
# =============================================================================

# pct_change is written explicitly with shift so it never fills a missing value.
for level_column, prefix in [("zhvi_usd", "zhvi"), ("zori_usd_month", "zori")]:
    previous_quarter = panel.groupby("county_fips")[level_column].shift(1)
    previous_year = panel.groupby("county_fips")[level_column].shift(4)
    previous_three_years = panel.groupby("county_fips")[level_column].shift(12)
    panel[f"{prefix}_growth_qoq_pct"] = 100 * (panel[level_column] / previous_quarter - 1)
    panel[f"{prefix}_growth_prev_1y_pct"] = 100 * (panel[level_column] / previous_year - 1)
    panel[f"{prefix}_growth_prev_3y_pct"] = 100 * (panel[level_column] / previous_three_years - 1)


# =============================================================================
# 4. CREATE BACKWARD-LOOKING ECONOMIC AND LABOR CHANGE FEATURES
# =============================================================================

# Rates are percentages, so their one-year changes are percentage points.
for level_column, feature_column in [
    ("nat_mortgage_rate_30y_pct", "mortgage_rate_change_prev_1y_pp"),
    ("cty_unemployment_rate_pct", "cty_unemployment_change_prev_1y_pp"),
    ("state_unemployment_rate_pct", "state_unemployment_change_prev_1y_pp"),
    ("nat_inflation_yoy_pct", "inflation_change_prev_1y_pp"),
]:
    previous_year = panel.groupby("county_fips")[level_column].shift(4)
    panel[feature_column] = panel[level_column] - previous_year

# A 1-year Treasury series is not in the source panel. Do not create a proxy.


# =============================================================================
# 5. CREATE DEMOGRAPHIC GROWTH AND AFFORDABILITY FEATURES
# =============================================================================

# ACS is annual data repeated across quarters. Only year-over-year changes are
# meaningful; missing 2020 and 2025 values remain missing in these features.
for level_column, feature_column in [
    ("population", "population_growth_prev_1y_pct"),
    ("median_household_income_usd", "income_growth_prev_1y_pct"),
    ("housing_units", "housing_units_growth_prev_1y_pct"),
    ("vacant_housing_units", "vacant_units_growth_prev_1y_pct"),
]:
    previous_year = panel.groupby("county_fips")[level_column].shift(4)
    panel[feature_column] = 100 * (panel[level_column] / previous_year - 1)

panel["price_to_rent_ratio"] = panel["zhvi_usd"] / (12 * panel["zori_usd_month"])
panel["price_to_income_ratio"] = panel["zhvi_usd"] / panel["median_household_income_usd"]
panel["rent_to_income_ratio"] = (12 * panel["zori_usd_month"]) / panel["median_household_income_usd"]


# =============================================================================
# 6. CREATE FORWARD-LOOKING LABELS (TARGETS ONLY; NEVER MODEL INPUTS)
# =============================================================================

for level_column, prefix in [("zhvi_usd", "zhvi"), ("zori_usd_month", "zori")]:
    next_year = panel.groupby("county_fips")[level_column].shift(-4)
    next_three_years = panel.groupby("county_fips")[level_column].shift(-12)
    panel[f"{prefix}_growth_ahead_1y_pct"] = 100 * (next_year / panel[level_column] - 1)
    panel[f"{prefix}_growth_ahead_3y_pct"] = 100 * (next_three_years / panel[level_column] - 1)


# =============================================================================
# 7. CHECK INVALID VALUES, MISSINGNESS, AND TARGET AVAILABILITY
# =============================================================================

numeric_columns = panel.select_dtypes(include="number").columns
assert not np.isinf(panel[numeric_columns].to_numpy(dtype=float)).any(), "Infinite value found."

for ratio_column in ["price_to_rent_ratio", "price_to_income_ratio", "rent_to_income_ratio"]:
    assert not (panel[ratio_column].dropna() <= 0).any(), f"Non-positive {ratio_column} found."

assert panel["zhvi_growth_ahead_1y_pct"].notna().sum() == 360
assert panel["zori_growth_ahead_1y_pct"].notna().sum() == 360
assert panel["zhvi_growth_ahead_3y_pct"].notna().sum() == 288
assert panel["zori_growth_ahead_3y_pct"].notna().sum() == 288

missingness = pd.DataFrame(
    {
        "variable_name": panel.columns,
        "missing_cells": [int(panel[column].isna().sum()) for column in panel.columns],
        "total_cells": len(panel),
        "missing_pct": [100 * panel[column].isna().mean() for column in panel.columns],
        "affected_counties": [
            ", ".join(sorted(panel.loc[panel[column].isna(), "county_name"].unique()))
            if panel[column].isna().any() else ""
            for column in panel.columns
        ],
        "affected_periods": [
            ", ".join(sorted(panel.loc[panel[column].isna(), "quarter_end_date"].astype(str).unique())[:12])
            + ("; additional periods" if panel[column].isna().sum() > 12 else "")
            if panel[column].isna().any() else ""
            for column in panel.columns
        ],
    }
)

feature_columns = [
    "zhvi_growth_qoq_pct", "zhvi_growth_prev_1y_pct", "zhvi_growth_prev_3y_pct",
    "zori_growth_qoq_pct", "zori_growth_prev_1y_pct", "zori_growth_prev_3y_pct",
    "mortgage_rate_change_prev_1y_pp", "cty_unemployment_change_prev_1y_pp",
    "state_unemployment_change_prev_1y_pp", "inflation_change_prev_1y_pp",
    "population_growth_prev_1y_pct", "income_growth_prev_1y_pct",
    "housing_units_growth_prev_1y_pct", "vacant_units_growth_prev_1y_pct",
    "price_to_rent_ratio", "price_to_income_ratio", "rent_to_income_ratio",
]
target_columns = [
    "zhvi_growth_ahead_1y_pct", "zori_growth_ahead_1y_pct",
    "zhvi_growth_ahead_3y_pct", "zori_growth_ahead_3y_pct",
]
quality_summary = panel[feature_columns + target_columns].describe().T.reset_index(names="variable_name")
quality_summary.to_csv(QUALITY_SUMMARY_FILE, index=False)


# =============================================================================
# 8. ORDER COLUMNS BY KEYS, SOURCE GROUP, FEATURES, THEN TARGETS
# =============================================================================

housing_columns = ["zhvi_usd", "zori_usd_month", "fhfa_hpi"]
demographic_columns = [
    "population", "median_household_income_usd", "median_household_income_moe_usd",
    "housing_units", "housing_units_moe", "vacant_housing_units",
    "vacant_housing_units_moe", "vacancy_rate_pct",
]
national_columns = [
    "nat_mortgage_rate_30y_pct", "nat_cpi_index", "nat_treasury_3y_pct",
    "nat_treasury_10y_pct", "nat_tbill_3m_pct", "nat_inflation_yoy_pct",
]
county_labor_columns = ["cty_unemployed", "cty_employed", "cty_labor_force", "cty_unemployment_rate_pct"]
state_labor_columns = ["state_unemployed", "state_employed", "state_labor_force", "state_unemployment_rate_pct"]
quality_columns = [
    "cty_n_months", "cty_partial_quarter", "nat_mortgage_rate_30y_pct_n_obs",
    "nat_cpi_index_n_obs", "nat_cpi_index_partial", "nat_treasury_3y_pct_n_obs",
    "nat_treasury_10y_pct_n_obs", "nat_inflation_yoy_pct_partial",
    "state_n_months", "state_partial_quarter",
]
column_order = (
    KEY_COLUMNS + housing_columns + demographic_columns + national_columns
    + county_labor_columns + state_labor_columns + quality_columns
    + feature_columns + target_columns
)
assert set(column_order) == set(panel.columns), "Feature-panel column order is incomplete."
panel = panel[column_order]


# =============================================================================
# 9. CREATE A FULL FEATURE DICTIONARY
# =============================================================================

source_dictionary = pd.read_csv(SOURCE_DICTIONARY_FILE).set_index("variable_name")
feature_definitions = {
    "zhvi_growth_qoq_pct": ("Housing momentum", "Derived from Zillow", "growth", "prev", "percent", "100 × (ZHVI_t / ZHVI_t-1 − 1)", "1 quarter", "Yes"),
    "zhvi_growth_prev_1y_pct": ("Housing momentum", "Derived from Zillow", "growth", "prev", "percent", "100 × (ZHVI_t / ZHVI_t-4 − 1)", "4 quarters", "Yes"),
    "zhvi_growth_prev_3y_pct": ("Housing momentum", "Derived from Zillow", "growth", "prev", "percent", "100 × (ZHVI_t / ZHVI_t-12 − 1)", "12 quarters", "Yes"),
    "zori_growth_qoq_pct": ("Housing momentum", "Derived from Zillow", "growth", "prev", "percent", "100 × (ZORI_t / ZORI_t-1 − 1)", "1 quarter", "Yes"),
    "zori_growth_prev_1y_pct": ("Housing momentum", "Derived from Zillow", "growth", "prev", "percent", "100 × (ZORI_t / ZORI_t-4 − 1)", "4 quarters", "Yes"),
    "zori_growth_prev_3y_pct": ("Housing momentum", "Derived from Zillow", "growth", "prev", "percent", "100 × (ZORI_t / ZORI_t-12 − 1)", "12 quarters", "Yes"),
    "mortgage_rate_change_prev_1y_pp": ("Economic changes", "Derived from FRED", "change", "prev", "percentage points", "Mortgage rate_t − Mortgage rate_t-4", "4 quarters", "Yes"),
    "cty_unemployment_change_prev_1y_pp": ("Economic changes", "Derived from BLS LAUS", "change", "prev", "percentage points", "County unemployment rate_t − rate_t-4", "4 quarters", "Yes"),
    "state_unemployment_change_prev_1y_pp": ("Economic changes", "Derived from BLS LAUS", "change", "prev", "percentage points", "Ohio unemployment rate_t − rate_t-4", "4 quarters", "Yes"),
    "inflation_change_prev_1y_pp": ("Economic changes", "Derived from FRED", "change", "prev", "percentage points", "Inflation_t − inflation_t-4", "4 quarters", "Yes"),
    "population_growth_prev_1y_pct": ("Demographic growth", "Derived from ACS", "growth", "prev", "percent", "100 × (population_t / population_t-4 − 1)", "4 quarters", "Conditional: ACS release lag"),
    "income_growth_prev_1y_pct": ("Demographic growth", "Derived from ACS", "growth", "prev", "percent", "100 × (income_t / income_t-4 − 1)", "4 quarters", "Conditional: ACS release lag"),
    "housing_units_growth_prev_1y_pct": ("Demographic growth", "Derived from ACS", "growth", "prev", "percent", "100 × (housing units_t / housing units_t-4 − 1)", "4 quarters", "Conditional: ACS release lag"),
    "vacant_units_growth_prev_1y_pct": ("Demographic growth", "Derived from ACS", "growth", "prev", "percent", "100 × (vacant units_t / vacant units_t-4 − 1)", "4 quarters", "Conditional: ACS release lag"),
    "price_to_rent_ratio": ("Affordability", "Derived from Zillow", "ratio", "current", "ratio", "ZHVI / (12 × monthly ZORI)", "Current quarter", "Yes"),
    "price_to_income_ratio": ("Affordability", "Derived from Zillow and ACS", "ratio", "current", "ratio", "ZHVI / median household income", "Current quarter", "Conditional: ACS release lag"),
    "rent_to_income_ratio": ("Affordability", "Derived from Zillow and ACS", "ratio", "current", "ratio", "(12 × monthly ZORI) / median household income", "Current quarter", "Conditional: ACS release lag"),
    "zhvi_growth_ahead_1y_pct": ("Prediction targets", "Derived from Zillow", "target", "ahead", "percent", "100 × (ZHVI_t+4 / ZHVI_t − 1)", "4 quarters ahead", "No — label only"),
    "zori_growth_ahead_1y_pct": ("Prediction targets", "Derived from Zillow", "target", "ahead", "percent", "100 × (ZORI_t+4 / ZORI_t − 1)", "4 quarters ahead", "No — label only"),
    "zhvi_growth_ahead_3y_pct": ("Prediction targets", "Derived from Zillow", "target", "ahead", "percent", "100 × (ZHVI_t+12 / ZHVI_t − 1)", "12 quarters ahead", "No — label only"),
    "zori_growth_ahead_3y_pct": ("Prediction targets", "Derived from Zillow", "target", "ahead", "percent", "100 × (ZORI_t+12 / ZORI_t − 1)", "12 quarters ahead", "No — label only"),
}

rows = []
for column in panel.columns:
    missing_note = f"{int(panel[column].isna().sum())} of {len(panel)} values missing."
    if column in feature_definitions:
        category, source, feature_type, direction, unit, formula, window, safe = feature_definitions[column]
        raw_or_engineered = "target" if feature_type == "target" else "engineered"
        comments = "Computed within county after sorting by year and quarter."
    else:
        source_row = source_dictionary.loc[column]
        category = "Keys" if column in KEY_COLUMNS else (
            "Housing source data" if column in housing_columns else
            "Demographic source data" if column in demographic_columns else
            "National economic source data" if column in national_columns else
            "County labor source data" if column in county_labor_columns else
            "Ohio labor source data" if column in state_labor_columns else "Source quality flags"
        )
        source = source_row["source"]
        feature_type = "time" if column in {"year", "quarter", "quarter_end_date"} else "level"
        direction = "current"
        unit = source_row["unit"]
        formula = source_row["transformation_or_rule"]
        window = "Current quarter"
        safe = "Conditional: ACS release lag" if source == "ACS" else "Yes"
        raw_or_engineered = "raw"
        comments = source_row["description"]
    rows.append((column, category, source, raw_or_engineered, feature_type, direction,
                 "Quarterly", unit, formula, window, safe, missing_note, comments))

feature_dictionary = pd.DataFrame(rows, columns=[
    "variable_name", "category", "source", "raw_or_engineered", "feature_type",
    "direction", "frequency", "unit", "exact_formula_or_definition",
    "lookback_or_forecast_window", "safe_at_prediction_time", "missingness_notes", "comments",
])
assert set(feature_dictionary["variable_name"]) == set(panel.columns)


# =============================================================================
# 10. WRITE ALL FEATURE-ENGINEERING OUTPUTS
# =============================================================================

panel.to_csv(FEATURE_PANEL_FILE, index=False)
missingness.to_csv(FEATURE_MISSINGNESS_FILE, index=False)
feature_dictionary.to_csv(FEATURE_DICTIONARY_FILE, index=False)

print(f"Wrote {FEATURE_PANEL_FILE.relative_to(ROOT)} with {len(panel)} rows and {len(panel.columns)} columns.")
print(f"Labeled rows: 1y ZHVI={panel['zhvi_growth_ahead_1y_pct'].notna().sum()}, "
      f"1y ZORI={panel['zori_growth_ahead_1y_pct'].notna().sum()}, "
      f"3y ZHVI={panel['zhvi_growth_ahead_3y_pct'].notna().sum()}, "
      f"3y ZORI={panel['zori_growth_ahead_3y_pct'].notna().sum()}.")

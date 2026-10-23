"""Build the Ohio nine-county quarterly master panel.

Run from the repository root:
    python src/build_master_county_quarter.py

The script intentionally follows the project's documented decisions:
* Zillow ZHVI and ZORI use March, June, September, and December observations.
* Annual FHFA HPI is repeated across the four quarters of its reference year.
* ACS source years 2020 and 2025 remain missing; no values are carried forward.
* National and Ohio state economic variables remain in the county-quarter panel.
"""

from pathlib import Path
import re

import pandas as pd


# =============================================================================
# 1. PROJECT SETTINGS AND FILE LOCATIONS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"

COUNTY_NAMES = {
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

KEY_COLUMNS = ["county_fips", "year", "quarter"]
COUNTY_FIPS = list(COUNTY_NAMES)
QUARTERS = pd.DataFrame({"quarter": [1, 2, 3, 4]})

ZHVI_FILE = PROCESSED / "Zillow" / "Ohio_9_Counties_ZHVI_Quarterly_2015_2025.csv"
ZORI_FILE = PROCESSED / "Zillow" / "Ohio_9_Counties_ZORI_Quarterly_2015_2025.csv"
FHFA_FILE = PROCESSED / "FHFA" / "Ohio_9_Counties_FHFA_HPI_Annual_2015_2025.csv"
ECONOMIC_LABOR_FILE = PROCESSED / "econ_labor_county_quarter.csv"
ECONOMIC_LABOR_DICTIONARY_FILE = PROCESSED / "econ_labor_variable_definitions.csv"

MERGED_DATA = PROCESSED / "merged data"
MASTER_FILE = MERGED_DATA / "master_county_quarter.csv"
MISSINGNESS_FILE = MERGED_DATA / "master_county_quarter_missingness.csv"
ACS_YEAR_FILE = PROCESSED / "ACS" / "acs_county_year.csv"
ACS_QUARTER_FILE = PROCESSED / "ACS" / "acs_county_quarter.csv"


# =============================================================================
# 2. BUILD THE COMPLETE COUNTY-QUARTER SKELETON
# =============================================================================

panel = pd.MultiIndex.from_product(
    [COUNTY_FIPS, range(2015, 2026), range(1, 5)],
    names=KEY_COLUMNS,
).to_frame(index=False)

panel["county_name"] = panel["county_fips"].map(COUNTY_NAMES)
panel["quarter_end_date"] = (
    pd.to_datetime(
        {
            "year": panel["year"],
            "month": panel["quarter"] * 3,
            "day": 1,
        }
    )
    + pd.offsets.MonthEnd(0)
)

assert len(panel) == 396, "The base panel must contain 9 counties x 44 quarters."
assert not panel.duplicated(KEY_COLUMNS).any(), "The base panel has duplicate keys."


# =============================================================================
# 3. RESHAPE ZILLOW ZHVI FROM WIDE FORMAT TO COUNTY-QUARTER FORMAT
# =============================================================================

zhvi_wide = pd.read_csv(ZHVI_FILE)
zhvi_date_columns = [
    column
    for column in zhvi_wide.columns
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(column))
]

zhvi = zhvi_wide.melt(
    id_vars=["RegionName", "StateCodeFIPS", "MunicipalCodeFIPS"],
    value_vars=zhvi_date_columns,
    var_name="quarter_end_date",
    value_name="zhvi_usd",
)

zhvi["quarter_end_date"] = pd.to_datetime(zhvi["quarter_end_date"])
zhvi["year"] = zhvi["quarter_end_date"].dt.year
zhvi["quarter"] = zhvi["quarter_end_date"].dt.quarter
zhvi["county_fips"] = (
    pd.to_numeric(zhvi["StateCodeFIPS"]).astype(int).astype(str).str.zfill(2)
    + pd.to_numeric(zhvi["MunicipalCodeFIPS"]).astype(int).astype(str).str.zfill(3)
)

assert set(zhvi["county_fips"]) == set(COUNTY_FIPS), "ZHVI counties do not match scope."
assert not zhvi.duplicated(KEY_COLUMNS).any(), "ZHVI has duplicate county-quarter keys."

zhvi = zhvi[KEY_COLUMNS + ["zhvi_usd"]]

panel = panel.merge(zhvi, on=KEY_COLUMNS, how="left", validate="one_to_one")
assert len(panel) == 396, "ZHVI merge changed the panel row count."


# =============================================================================
# 4. RESHAPE ZILLOW ZORI FROM WIDE FORMAT TO COUNTY-QUARTER FORMAT
# =============================================================================

zori_wide = pd.read_csv(ZORI_FILE)
zori_date_columns = [
    column
    for column in zori_wide.columns
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(column))
]

zori = zori_wide.melt(
    id_vars=["RegionName", "StateCodeFIPS", "MunicipalCodeFIPS"],
    value_vars=zori_date_columns,
    var_name="quarter_end_date",
    value_name="zori_usd_month",
)

zori["quarter_end_date"] = pd.to_datetime(zori["quarter_end_date"])
zori["year"] = zori["quarter_end_date"].dt.year
zori["quarter"] = zori["quarter_end_date"].dt.quarter
zori["county_fips"] = (
    pd.to_numeric(zori["StateCodeFIPS"]).astype(int).astype(str).str.zfill(2)
    + pd.to_numeric(zori["MunicipalCodeFIPS"]).astype(int).astype(str).str.zfill(3)
)

assert set(zori["county_fips"]) == set(COUNTY_FIPS), "ZORI counties do not match scope."
assert not zori.duplicated(KEY_COLUMNS).any(), "ZORI has duplicate county-quarter keys."

zori = zori[KEY_COLUMNS + ["zori_usd_month"]]

panel = panel.merge(zori, on=KEY_COLUMNS, how="left", validate="one_to_one")
assert len(panel) == 396, "ZORI merge changed the panel row count."


# =============================================================================
# 5. EXPAND ANNUAL FHFA HPI TO THE FOUR QUARTERS OF EACH REFERENCE YEAR
# =============================================================================

fhfa_wide = pd.read_csv(FHFA_FILE)
fhfa_year_columns = [str(year) for year in range(2015, 2026)]

fhfa = fhfa_wide.melt(
    id_vars=["County", "FIPS"],
    value_vars=fhfa_year_columns,
    var_name="year",
    value_name="fhfa_hpi",
)

fhfa["county_fips"] = pd.to_numeric(fhfa["FIPS"]).astype(int).astype(str).str.zfill(5)
fhfa["year"] = fhfa["year"].astype(int)
fhfa = fhfa.merge(QUARTERS, how="cross")

assert set(fhfa["county_fips"]) == set(COUNTY_FIPS), "FHFA counties do not match scope."
assert not fhfa.duplicated(KEY_COLUMNS).any(), "FHFA has duplicate county-quarter keys."

fhfa = fhfa[KEY_COLUMNS + ["fhfa_hpi"]]

panel = panel.merge(fhfa, on=KEY_COLUMNS, how="left", validate="one_to_one")
assert len(panel) == 396, "FHFA merge changed the panel row count."


# =============================================================================
# 6. BUILD COUNTY-YEAR ACS DATA FROM RAW COUNTY DOWNLOADS
# =============================================================================

acs_raw_files = sorted((RAW / "ACS").glob("acs_*_all_counties.csv"))
if not acs_raw_files:
    raise FileNotFoundError(
        "No county-level ACS raw files found. Merge Person 2's raw ACS files before running this script."
    )

acs_year_tables = []

for raw_file in acs_raw_files:
    year_match = re.search(r"acs_(\d{4})_all_counties", raw_file.stem)
    if year_match is None:
        raise ValueError(f"Could not read an ACS year from {raw_file.name}.")

    acs_year = int(year_match.group(1))
    acs_raw = pd.read_csv(raw_file, header=1, dtype={"state": str, "county": str})

    acs_raw["county_fips"] = acs_raw["state"].str.zfill(2) + acs_raw["county"].str.zfill(3)
    acs = acs_raw.loc[acs_raw["county_fips"].isin(COUNTY_FIPS)].copy()

    if set(acs["county_fips"]) != set(COUNTY_FIPS):
        missing_fips = sorted(set(COUNTY_FIPS) - set(acs["county_fips"]))
        raise ValueError(f"{raw_file.name} is missing required counties: {missing_fips}")

    census_columns = [
        "B01003_001E",
        "B19013_001E", "B19013_001M",
        "B25001_001E", "B25001_001M",
        "B25002_003E", "B25002_003M",
        "B25002_001E",
    ]

    for census_column in census_columns:
        acs[census_column] = pd.to_numeric(acs[census_column], errors="coerce")
        acs[census_column] = acs[census_column].replace(-555555555, pd.NA)

    acs["county_name"] = acs["county_fips"].map(COUNTY_NAMES)
    acs["year"] = acs_year
    acs["population"] = acs["B01003_001E"]
    acs["median_household_income_usd"] = acs["B19013_001E"]
    acs["median_household_income_moe_usd"] = acs["B19013_001M"]
    acs["housing_units"] = acs["B25001_001E"]
    acs["housing_units_moe"] = acs["B25001_001M"]
    acs["vacant_housing_units"] = acs["B25002_003E"]
    acs["vacant_housing_units_moe"] = acs["B25002_003M"]
    # B25001 total units and B25002 total units are identical for every
    # project county-year. Keep B25001 as the single canonical housing-units
    # measure and use it as the vacancy-rate denominator.
    assert (acs["B25001_001E"] == acs["B25002_001E"]).all(), (
        "ACS total housing-unit variables disagree; do not remove the occupancy-table total."
    )
    acs["vacancy_rate_pct"] = 100 * acs["vacant_housing_units"] / acs["housing_units"]

    acs_year_tables.append(
        acs[
            [
                "county_fips", "county_name", "year", "population",
                "median_household_income_usd", "median_household_income_moe_usd",
                "housing_units", "housing_units_moe", "vacant_housing_units",
                "vacant_housing_units_moe", "vacancy_rate_pct",
            ]
        ]
    )

acs_year = pd.concat(acs_year_tables, ignore_index=True).sort_values(["county_fips", "year"])

assert not acs_year.duplicated(["county_fips", "year"]).any(), "ACS has duplicate county-year keys."
assert set(acs_year["year"]) == {2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024}

ACS_YEAR_FILE.parent.mkdir(parents=True, exist_ok=True)
acs_year.to_csv(ACS_YEAR_FILE, index=False)


# =============================================================================
# 7. REPEAT OBSERVED ACS ANNUAL VALUES ACROSS FOUR QUARTERS
# =============================================================================

acs_quarter = acs_year.merge(QUARTERS, how="cross")

assert not acs_quarter.duplicated(KEY_COLUMNS).any(), "ACS has duplicate county-quarter keys."

acs_quarter.to_csv(ACS_QUARTER_FILE, index=False)

acs_for_merge = acs_quarter.drop(columns=["county_name"])
panel = panel.merge(acs_for_merge, on=KEY_COLUMNS, how="left", validate="one_to_one")
assert len(panel) == 396, "ACS merge changed the panel row count."


# =============================================================================
# 8. MERGE PERSON 3 ECONOMIC AND LABOR DATA
# =============================================================================

economic_labor = pd.read_csv(ECONOMIC_LABOR_FILE)
economic_labor["county_fips"] = pd.to_numeric(economic_labor["county_fips"]).astype(int).astype(str).str.zfill(5)
economic_labor["year"] = economic_labor["year"].astype(int)
economic_labor["quarter"] = economic_labor["quarter"].astype(int)

assert len(economic_labor) == 396, "Economic/labor data must contain 396 county-quarter rows."
assert set(economic_labor["county_fips"]) == set(COUNTY_FIPS), "Economic/labor counties do not match scope."
assert not economic_labor.duplicated(KEY_COLUMNS).any(), "Economic/labor data has duplicate keys."

economic_for_merge = economic_labor.drop(columns=["county_name"], errors="ignore")
panel = panel.merge(economic_for_merge, on=KEY_COLUMNS, how="left", validate="one_to_one")
assert len(panel) == 396, "Economic/labor merge changed the panel row count."
assert not panel.duplicated(KEY_COLUMNS).any(), "The final panel has duplicate county-quarter keys."


# =============================================================================
# 9. ORDER THE SOURCE PANEL BY KEYS, DATA SOURCE, AND QUALITY FLAGS
# =============================================================================

# Keep this file as the cleaned source panel. Feature engineering and targets
# are built separately in build_master_county_quarter_features.py so no
# forward-looking values can be mistaken for model inputs.
panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)

# =============================================================================
# 10. REMOVE EMPTY, DUPLICATE, AND CONSTANT QUALITY FIELDS
# =============================================================================

# These fields either contain no observed values, duplicate another retained
# field, or are constant across the current 396-row panel. Their definitions
# remain documented in source notes when needed, but they do not belong in the
# EDA-ready master table.
columns_to_remove = [
    "cty_any_preliminary",
    "cty_missing_reason",
    "hpi_index_basis",
    "hpi_source_frequency",
    "nat_mortgage_rate_30y_pct_partial",
    "nat_tbill_3m_pct_n_obs",
    "nat_tbill_3m_pct_partial",
    "nat_treasury_1y_pct_partial",
    "nat_treasury_10y_pct_partial",
    "nat_treasury_3y_pct_partial",
    "state_any_preliminary",
    "state_missing_reason",
]

panel = panel.drop(columns=columns_to_remove, errors="ignore")

assert not panel.isna().all().any(), "The EDA-ready master table has an all-missing column."
assert not panel.duplicated(KEY_COLUMNS).any(), "The cleaned master table has duplicate keys."

# The written order makes the CSV easy to inspect: identifying keys first,
# then source data grouped by provider, followed by source-quality flags.
column_order = [
    "county_fips", "county_name", "year", "quarter", "quarter_end_date",
    "zhvi_usd", "zori_usd_month", "fhfa_hpi",
    "population", "median_household_income_usd", "median_household_income_moe_usd",
    "housing_units", "housing_units_moe", "vacant_housing_units",
    "vacant_housing_units_moe", "vacancy_rate_pct",
    "nat_mortgage_rate_30y_pct", "nat_cpi_index", "nat_treasury_1y_pct", "nat_treasury_3y_pct",
    "nat_treasury_10y_pct", "nat_tbill_3m_pct", "nat_inflation_yoy_pct",
    "cty_unemployed", "cty_employed", "cty_labor_force", "cty_unemployment_rate_pct",
    "state_unemployed", "state_employed", "state_labor_force", "state_unemployment_rate_pct",
    "cty_n_months", "cty_partial_quarter", "nat_mortgage_rate_30y_pct_n_obs",
    "nat_cpi_index_n_obs", "nat_cpi_index_partial", "nat_treasury_1y_pct_n_obs", "nat_treasury_3y_pct_n_obs",
    "nat_treasury_10y_pct_n_obs", "nat_inflation_yoy_pct_partial",
    "state_n_months", "state_partial_quarter",
]
assert set(column_order) == set(panel.columns), "The source-panel column order is incomplete."
panel = panel[column_order]


# =============================================================================
# 11. WRITE THE MASTER PANEL, MISSINGNESS REPORT, AND DATA DICTIONARY
# =============================================================================

MERGED_DATA.mkdir(parents=True, exist_ok=True)
METADATA.mkdir(parents=True, exist_ok=True)

panel.to_csv(MASTER_FILE, index=False)

quality_columns = {
    "county_fips", "county_name", "year", "quarter", "quarter_end_date",
}
quality_suffixes = ("_moe", "_n_obs", "_n_months", "_partial", "_partial_quarter", "_missing_reason")

data_columns = [
    column
    for column in panel.columns
    if column not in quality_columns and not column.endswith(quality_suffixes)
]

missingness = pd.DataFrame(
    {
        "variable": data_columns,
        "missing_cells": [int(panel[column].isna().sum()) for column in data_columns],
        "total_cells": len(panel),
        "missing_pct": [100 * panel[column].isna().mean() for column in data_columns],
    }
)
missingness.to_csv(MISSINGNESS_FILE, index=False)

dictionary_columns = [
    "variable_name", "description", "unit", "source", "geographic_level",
    "native_frequency", "panel_frequency", "transformation_or_rule", "source_url",
]

dictionary = pd.DataFrame(
    [
        ("county_fips", "Five-digit Ohio county FIPS code.", "code", "Project key", "county", "Not applicable", "Quarterly", "Exact source county identifier.", ""),
        ("county_name", "Ohio county name without the word County.", "text", "Project key", "county", "Not applicable", "Quarterly", "Mapped from county_fips.", ""),
        ("year", "Calendar year of the quarter.", "year", "Project key", "county", "Not applicable", "Quarterly", "2015 through 2025.", ""),
        ("quarter", "Calendar quarter number.", "quarter", "Project key", "county", "Not applicable", "Quarterly", "1 through 4.", ""),
        ("quarter_end_date", "Last calendar day of the panel quarter.", "date", "Derived", "county", "Not applicable", "Quarterly", "Used as the panel time label.", ""),
        ("zhvi_usd", "Zillow Home Value Index at the quarter-end month.", "USD", "Zillow", "county", "Monthly", "Quarterly", "March, June, September, and December month-end observations.", ""),
        ("zori_usd_month", "Zillow Observed Rent Index at the quarter-end month.", "USD/month", "Zillow", "county", "Monthly", "Quarterly", "March, June, September, and December month-end observations.", ""),
        ("fhfa_hpi", "FHFA county House Price Index.", "index points", "FHFA", "county", "Annual", "Quarterly", "Annual source value, indexed to 2000=100, repeated across Q1 through Q4.", ""),
        ("population", "ACS total population estimate.", "persons", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; missing in 2020 and 2025.", ""),
        ("median_household_income_usd", "ACS median household income estimate.", "USD", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; missing in 2020 and 2025.", ""),
        ("median_household_income_moe_usd", "Margin of error for ACS median household income.", "USD", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; source sentinel values become missing.", ""),
        ("housing_units", "ACS housing-unit estimate.", "housing units", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; missing in 2020 and 2025.", ""),
        ("housing_units_moe", "Margin of error for ACS housing units.", "housing units", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; source sentinel values become missing.", ""),
        ("vacant_housing_units", "ACS vacant-housing-unit estimate.", "housing units", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; missing in 2020 and 2025.", ""),
        ("vacant_housing_units_moe", "Margin of error for ACS vacant housing units.", "housing units", "ACS", "county", "Annual", "Quarterly", "Repeated across all quarters of observed years; source sentinel values become missing.", ""),
        ("vacancy_rate_pct", "Vacant ACS units divided by total ACS housing units.", "percent", "Derived from ACS", "county", "Annual", "Quarterly", "100 x vacant_housing_units / housing_units; repeated across all quarters of observed years.", ""),
    ],
    columns=dictionary_columns,
)

economic_dictionary = pd.read_csv(ECONOMIC_LABOR_DICTIONARY_FILE)
economic_dictionary = economic_dictionary[
    ~economic_dictionary["variable"].astype(str).str.contains("\\*", regex=False)
].copy()
economic_dictionary = economic_dictionary[
    economic_dictionary["variable"].isin(panel.columns)
    & ~economic_dictionary["variable"].isin(dictionary["variable_name"])
].copy()
economic_dictionary = economic_dictionary.rename(
    columns={"variable": "variable_name", "quarterly_rule": "transformation_or_rule"}
)
economic_dictionary["native_frequency"] = "Monthly or daily"
# DGS1 is a daily FRED series, so record its exact frequency in the master
# dictionary rather than the broader default used by the older source rows.
economic_dictionary.loc[
    economic_dictionary["variable_name"] == "nat_treasury_1y_pct",
    "native_frequency",
] = "Daily"
economic_dictionary["panel_frequency"] = "Quarterly"
economic_dictionary = economic_dictionary[
    [
        "variable_name", "description", "unit", "source", "geo_level",
        "native_frequency", "panel_frequency", "transformation_or_rule", "source_url",
    ]
].rename(columns={"geo_level": "geographic_level"})

quality_dictionary = pd.DataFrame(
    [
        ("cty_n_months", "Number of BLS county source months used in the quarterly value.", "count", "BLS LAUS", "county", "Monthly", "Quarterly", "Three normally; two for the documented 2025Q4 partial quarter.", "https://www.bls.gov/lau/"),
        ("cty_partial_quarter", "Whether the county quarterly value uses an incomplete source quarter.", "boolean", "BLS LAUS", "county", "Monthly", "Quarterly", "True for the documented 2025Q4 partial quarter.", "https://www.bls.gov/lau/"),
        ("state_n_months", "Number of Ohio state BLS source months used in the quarterly value.", "count", "BLS LAUS", "state", "Monthly", "Quarterly", "Repeated across counties by design.", "https://www.bls.gov/lau/"),
        ("state_partial_quarter", "Whether the Ohio state quarterly value uses an incomplete source quarter.", "boolean", "BLS LAUS", "state", "Monthly", "Quarterly", "Repeated across counties by design.", "https://www.bls.gov/lau/"),
        ("nat_mortgage_rate_30y_pct_n_obs", "Number of weekly mortgage-rate observations used in the quarter.", "count", "FRED", "national", "Weekly", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/MORTGAGE30US"),
        ("nat_cpi_index_n_obs", "Number of monthly CPI observations used in the quarter.", "count", "FRED", "national", "Monthly", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/CPIAUCSL"),
        ("nat_cpi_index_partial", "Whether the quarterly CPI value is based on an incomplete source period.", "boolean", "FRED", "national", "Monthly", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/CPIAUCSL"),
        ("nat_treasury_1y_pct_n_obs", "Number of daily 1-year Treasury observations used in the quarter.", "count", "FRED", "national", "Daily", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/DGS1"),
        ("nat_treasury_3y_pct_n_obs", "Number of daily 3-year Treasury observations used in the quarter.", "count", "FRED", "national", "Daily", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/DGS3"),
        ("nat_treasury_10y_pct_n_obs", "Number of daily 10-year Treasury observations used in the quarter.", "count", "FRED", "national", "Daily", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/DGS10"),
        ("nat_inflation_yoy_pct_partial", "Whether the quarterly year-over-year inflation value uses an incomplete source period.", "boolean", "Derived from FRED", "national", "Monthly", "Quarterly", "Repeated across counties by design.", "https://fred.stlouisfed.org/series/CPIAUCSL"),
    ],
    columns=dictionary_columns,
)

dictionary = pd.concat([dictionary, economic_dictionary, quality_dictionary], ignore_index=True)
dictionary = dictionary.drop_duplicates("variable_name").sort_values("variable_name")

assert set(dictionary["variable_name"]) == set(panel.columns), "The master dictionary is missing a panel column."

dictionary.to_csv(METADATA / "master_data_dictionary.csv", index=False)

print(f"Wrote {MASTER_FILE.relative_to(ROOT)} with {len(panel)} rows and {len(panel.columns)} columns.")
print(f"Wrote {MISSINGNESS_FILE.relative_to(ROOT)}.")
print(f"Wrote {(METADATA / 'master_data_dictionary.csv').relative_to(ROOT)}.")

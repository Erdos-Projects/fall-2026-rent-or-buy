# County-Quarter Data Integration

## Canonical Outputs

Run the integration pipeline from the repository root:

```bash
python src/build_master_county_quarter.py
```

It produces the following canonical files:

- `data/processed/ACS/acs_county_year.csv` **Person 2's cleaned ACS original county × year**
- `data/processed/ACS/acs_county_quarter.csv` **The ACS county-year values repeated across Q1–Q4**
- `data/processed/merged data/master_county_quarter.csv` **The cleaned source panel, arranged by keys, source, and quality flags**
- `data/processed/merged data/master_county_quarter_missingness.csv` **A variable-by-variable missingness audit.**
- `data/metadata/master_data_dictionary.csv` **The documentation table for every column in the master dataset.**

The master panel has one row for every `county_fips × year × quarter` combination:
nine Ohio counties, 2015Q1–2025Q4, or 396 rows.

## Merge Rules

- **Zillow ZHVI and ZORI:** March, June, September, and December month-end
  observations are the Q1, Q2, Q3, and Q4 values.
- **FHFA HPI:** the supplied annual county value, indexed to 2000=100, is
  repeated in Q1–Q4 of its reference year.
- **ACS:** each observed annual estimate is repeated within its own four
  quarters. All ACS fields remain missing for 2020 and 2025; the pipeline does
  not carry values forward or impute them.
- **Economic and labor variables:** Person 3's national (`nat_*`) and Ohio
  state (`state_*`) fields repeat across counties by design. County labor
  fields (`cty_*`) vary by county and quarter. National fields include
  `nat_treasury_1y_pct`, the quarterly mean FRED DGS1 yield used as the
  project's default low-risk return benchmark.

## EDA-Ready Master Table

The master table excludes empty columns, exact duplicate ACS columns, and
constant source metadata. In particular, it excludes `population_moe`,
`occupancy_total_units`, `occupancy_total_units_moe`, `acs_reference_year`,
`acs_carried_forward`, `hpi_index_basis`, and `hpi_source_frequency`.

It retains fields that identify an actual source limitation. For example,
`cty_partial_quarter`, `state_partial_quarter`, `nat_cpi_index_partial`, and
`nat_inflation_yoy_pct_partial` are `True` for the nine 2025Q4 rows because
their underlying quarterly values use only November and December observations.

`nat_treasury_1y_pct` and its daily-observation count
`nat_treasury_1y_pct_n_obs` are retained. The corresponding source partial
flag is excluded because it is always `False` in the current 2015Q1–2025Q4
panel.

## Feature Engineering

The source panel intentionally contains only current source variables and source-quality fields. `src/build_master_county_quarter_features.py` creates the separate model-ready feature panel and documents one-year and three-year historical features and prediction targets in [checkpoint2_feature_engineering.md](checkpoint2_feature_engineering.md). This keeps forward-looking labels out of the source panel and prevents them from being treated as predictors.

## Checks Performed by the Script

The script stops if an input does not contain the nine required county FIPS
codes, if a source has duplicate merge keys, or if a merge changes the 396-row
panel. It writes a missingness report alongside the master data and verifies
that the master data dictionary contains every output column.

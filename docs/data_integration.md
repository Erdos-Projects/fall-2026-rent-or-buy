# County-Quarter Data Integration

## Canonical Outputs

Run the integration pipeline from the repository root:

```bash
python src/build_master_county_quarter.py
```

It produces the following canonical files:

- `data/processed/ACS/acs_county_year.csv` **Person 2's cleaned ACS original county × year**
- `data/processed/ACS/acs_county_quarter.csv` **The ACS county-year values repeated across Q1–Q4**
- `data/processed/master_county_quarter.csv` **The final analysis-ready dataset**
- `data/processed/master_county_quarter_missingness.csv` **A variable-by-variable missingness audit.**
- `data/metadata/master_data_dictionary.csv` **The documentation table for every column in the master dataset.**

The master panel has one row for every `county_fips × year × quarter` combination:
nine Ohio counties, 2015Q1–2025Q4, or 396 rows.

## Merge Rules

- **Zillow ZHVI and ZORI:** March, June, September, and December month-end
  observations are the Q1, Q2, Q3, and Q4 values.
- **FHFA HPI:** the supplied annual county value is repeated in Q1–Q4 of its
  reference year. `hpi_source_frequency` remains `annual`.
- **ACS:** each observed annual estimate is repeated within its own four
  quarters. All ACS fields remain missing for 2020 and 2025; the pipeline does
  not carry values forward or impute them.
- **Economic and labor variables:** Person 3's national (`nat_*`) and Ohio
  state (`state_*`) fields repeat across counties by design. County labor
  fields (`cty_*`) vary by county and quarter.

## Derived Fields

- `price_to_income = zhvi_usd / median_household_income_usd`
- `price_to_rent = zhvi_usd / (12 × zori_usd_month)`
- `zhvi_growth_3y_pct` and `zori_growth_3y_pct` use the same county's value
  twelve quarters later.

The three-year-ahead outcomes are missing for 2023Q1–2025Q4 because the panel
ends in 2025Q4. This indicates an unavailable future outcome, not a missing
current Zillow observation.

## Checks Performed by the Script

The script stops if an input does not contain the nine required county FIPS
codes, if a source has duplicate merge keys, or if a merge changes the 396-row
panel. It writes a missingness report alongside the master data and verifies
that the master data dictionary contains every output column.

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

## Modeling Variables and Prediction Targets

The machine-learning task uses information available for county `c` in quarter
`t` to predict housing-market growth twelve quarters later. The prediction
unit is therefore one `county_fips × year × quarter` row.

**Predictor groups** include:

- **Current housing conditions:** `zhvi_usd`, `zori_usd_month`, `fhfa_hpi`,
  `price_to_income`, and `price_to_rent`.
- **County demographics:** `population`, `median_household_income_usd`,
  `housing_units`, and `vacancy_rate_pct`. ACS margins of error are retained
  for data-quality assessment, rather than treated as core predictors by
  default.
- **National economic context:** mortgage rate, CPI, inflation, Treasury and
  Treasury-bill rate columns with the `nat_*` prefix.
- **Labor-market conditions:** Ohio statewide `state_*` employment, labor
  force, and unemployment measures, plus county-specific `cty_*` measures.

The two supervised-learning targets are three-year percentage growth in Zillow
home values and rents:

\[
\text{ZHVI growth}_{c,t \rightarrow t+12}
= 100\left(\frac{\text{ZHVI}_{c,t+12}}{\text{ZHVI}_{c,t}} - 1\right)
\]

\[
\text{ZORI growth}_{c,t \rightarrow t+12}
= 100\left(\frac{\text{ZORI}_{c,t+12}}{\text{ZORI}_{c,t}} - 1\right)
\]

These are stored as `zhvi_growth_3y_pct` and `zori_growth_3y_pct`.
After estimating either growth rate, the implied future level for the
rent-versus-buy application is:

\[
\widehat{V}_{c,t+12}=V_{c,t}\left(1+
\frac{\widehat{g}_{c,t \rightarrow t+12}}{100}\right),
\]

where \(V\) is ZHVI or ZORI and \(\widehat{g}\) is the model prediction.

The source table aligns ACS estimates with their reference year. Before a
real-time forecasting exercise, the modeling workflow must lag ACS features
to their public release dates so that a model never uses data unavailable at
the prediction date.

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

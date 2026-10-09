# Economic & labor data (Person 3): FRED + BLS

Unit: **county × quarter**, 2015Q1–2025Q4, 9 Ohio counties (396 rows).
Code: [`src/econ_labor_pipeline.py`](../src/econ_labor_pipeline.py).

## How to run

```bash
pip install pandas requests
# optional but recommended: a free BLS key from https://data.bls.gov/registrationEngine/
export BLS_API_KEY=your_key        # Windows PowerShell: $env:BLS_API_KEY="your_key"
python src/econ_labor_pipeline.py          # download raw files + build processed tables
python src/econ_labor_pipeline.py build    # rebuild processed tables from saved raw files only
```

FRED needs no key; the script uses the public `fredgraph.csv` download. Without a BLS key it uses the v1 API in smaller batches. Never commit API keys.

## Variables

Every variable carries a geographic-level prefix: `nat_` = national, `state_` = Ohio, `cty_` = county.

| Variable | Level | Source / series | Native frequency | Quarterly rule |
|---|---|---|---|---|
| `nat_mortgage_rate_30y_pct` | national | FRED `MORTGAGE30US` | weekly | mean of weeks dated in quarter (≥12 obs) |
| `nat_cpi_index` | national | FRED `CPIAUCSL` (SA, 1982–84=100) | monthly | mean of available months (≥2 of 3) |
| `nat_inflation_yoy_pct` | national | derived from CPI | — | 100 × (CPI_q / CPI_q−4 − 1) |
| `nat_treasury_3y_pct` | national | FRED `DGS3` | daily | mean of trading days (≥55 obs) |
| `nat_treasury_10y_pct` | national | FRED `DGS10` | daily | mean of trading days (≥55 obs) |
| `nat_tbill_3m_pct` | national | FRED `TB3MS` | monthly | mean of available months (≥2 of 3) |
| `state_unemployed`, `state_employed`, `state_labor_force` | state | BLS LAUS `LASST390000000000004`, `…05`, `…06` (SA) | monthly | mean of available months (≥2 of 3) |
| `state_unemployment_rate_pct` | state | derived | — | 100 × unemployed / labor force |
| `cty_unemployed`, `cty_employed`, `cty_labor_force` | county | BLS LAUS `LAUCN<fips>00000000{04,05,06}` (NSA) | monthly | mean of available months (≥2 of 3) |
| `cty_unemployment_rate_pct` | county | derived | — | 100 × unemployed / labor force |

The full definition table is written to `data/processed/econ_labor_variable_definitions.csv`.

**Low-risk return benchmark (team decision, 2026-10-08): `nat_treasury_3y_pct`.** The project predicts 3 years ahead, so the 3-year Treasury yield matches the horizon. The 10-year and 3-month rates stay in the table for sensitivity checks only.

## Counties

| FIPS | County | FIPS | County |
|---|---|---|---|
| 39017 | Butler | 39113 | Montgomery |
| 39035 | Cuyahoga | 39151 | Stark |
| 39049 | Franklin | 39153 | Summit |
| 39061 | Hamilton | 39165 | Warren |
| 39095 | Lucas | | |

## Rules

- **Raw files are never edited.** They are saved under `data/raw/{FRED,BLS}/<retrieved_date>/`. URL, retrieval time and SHA-256 are recorded in `data/raw/raw_manifest_econ_labor.csv`.
- **Two-month quarters are accepted and flagged (team decision, 2026-10-08).** Monthly series (CPI, 3-month bill, LAUS counts) use the mean of the available months if at least 2 of 3 exist. Such values carry `*_partial = True` (national) or `*_partial_quarter = True` (state, county). Weekly and daily series keep their thresholds (≥12 weeks, ≥55 trading days). Quarters below the threshold stay missing. Observation counts (`*_n_obs`, `*_n_months`) and `*_missing_reason` are kept.
- **Unemployment rates come from counts.** They are calculated from quarterly unemployed and labor-force counts, never by averaging monthly rates.
- **Raw downloads include 2014.** This allows 2015 year-over-year inflation. Processed outputs start in 2015Q1.
- **No imputation.** A two-month quarter is an average of observed months, not a filled-in value.

## Known limitations

- **County data are not seasonally adjusted.** BLS does not seasonally adjust LAUS county data. Quarterly means smooth some, but not all, seasonality. The state series is seasonally adjusted.
- **Employment counts are residents, not jobs.** LAUS employment counts employed residents, not payroll jobs located in the county.
- **Recent months are preliminary and get revised.** They are flagged in `*_any_preliminary`. BLS revises LAUS data annually, so the retrieval date matters.
- **2025Q4 rests on two months (November and December).** BLS did not publish October 2025 CPI or LAUS values because of the October–November 2025 federal government shutdown ([BLS FAQ](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)). The team chose to keep 2025Q4 as a flagged two-month average rather than leave it blank. This is a potential limitation of the model:
    - The 2025Q4 value is not strictly comparable with other quarters. County data are not seasonally adjusted, so dropping October can shift the quarterly mean.
    - 2025Q4 inflation compares a November–December average with a full October–December 2024 average.
    - Use the `*_partial` flags to run a sensitivity check that leaves 2025Q4 out. Because 2025Q4 has no 3-year-ahead outcome, it affects prediction inputs, not model training.
- **National variables are the same for all counties.** They add only 44 distinct values to the panel.

## Coverage and missingness (run of 2026-10-08)

Raw data retrieved 2026-10-08: FRED fredgraph CSV, and BLS API v1 without a key in 4 batches. The combined table has 396 rows (9 counties × 44 quarters), with 0 duplicate `county × quarter` keys.

| Variable group | Observed | Missing | Reason |
|---|---|---|---|
| Mortgage rate, 3y/10y Treasury, 3m T-bill | 2015Q1–2025Q4 (44/44) | none | — |
| CPI, YoY inflation | 2015Q1–2025Q4 (44/44) | none | 2025Q4 is a flagged 2-month quarter (October 2025 not published) |
| Ohio labor (state) | 2015Q1–2025Q4 (44/44) | none | 2025Q4 is a flagged 2-month quarter |
| County labor, 9 counties | 2015Q1–2025Q4 (396/396 cells) | none | 2025Q4 is a flagged 2-month quarter for all 9 counties |

Checks passed:

- employed + unemployed = labor force for every county-quarter
- unemployment rate = unemployed / labor force
- mortgage rate quarters have 12–14 weekly observations
- Treasury quarters have 61–64 trading days

Plausibility:

- The COVID shock appears in 2020Q2: Ohio unemployment rate 13.4%, Lucas 19.0%, Cuyahoga 17.6%.
- CPI inflation peaks at 8.6% in 2022Q2.
- The 30-year mortgage rate peaks at 7.3% in 2023Q4.

Labor force 2015Q1 → 2025Q3:

- Largest growth: Franklin (+14%) and Warren (+21%)
- Small declines: Lucas and Stark

No BLS preliminary flags appear in this pull.

Under the original complete-quarter rule, 2025Q4 CPI, inflation and labor values were missing. After the team decision of 2026-10-08, the table has no missing cells; 9 rows (2025Q4) are flagged as partial quarters.

## Usage terms

- BLS LAUS and CPI, and U.S. Treasury rates, are U.S. government data in the public domain. Cite the source.
- `MORTGAGE30US` is Freddie Mac Primary Mortgage Market Survey data distributed through FRED. Cite Freddie Mac and FRED, and check the FRED series notes before redistributing the raw file.

## Outputs

| File | Rows |
|---|---|
| `data/processed/econ_labor_county_quarter.csv` (main deliverable) | 396 |
| `data/processed/FRED/national_quarterly.csv` | 44; national input table used to build the county-quarter panel |
| `data/processed/BLS/state_labor_quarterly.csv` | 44; Ohio reference table used to build the county-quarter panel |
| `data/processed/BLS/county_labor_quarterly.csv` | 396 |
| `data/processed/econ_labor_variable_definitions.csv` | — |
| `data/processed/econ_labor_coverage.csv` (missing cells per variable) | — |

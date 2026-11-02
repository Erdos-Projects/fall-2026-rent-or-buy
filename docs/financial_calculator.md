# Financial calculator: modes and 2025 forecast extension

[Open the calculator](https://htmlpreview.github.io/?https://github.com/Erdos-Projects/fall-2026-rent-or-buy/blob/repository-clean-up/financial_calculator.html) · [Project overview](../README.md)

The calculator compares **one-year ending wealth** from buying versus renting
and investing, using the same starting cash and monthly budget. All decision
quarters are historical project snapshots. This is **not a live forecast**.

## What the two modes mean

| Mode | Housing growth used | Available decision quarters | Interpretation |
|---|---|---|---|
| **Model forecast** | Predicted one-year ZHVI and ZORI growth | 2023Q1–2025Q4 | A financial scenario conditional on the models' predictions and the user's assumptions |
| **Historical outcome** | Observed one-year ZHVI and ZORI growth | 2015Q1–2024Q4 | A hindsight scenario using realized housing growth; costs and investment returns are still assumed |

For example, **2025Q1 is the decision quarter; 2026Q1 is the forecast endpoint**.
All four 2025 origins have forecasts. Their 2026 endpoints are outside the
project's observed panel, so they have **no observed outcome or accuracy score**.
Switching to Historical outcome therefore removes 2025 from the quarter menu.
It does not substitute predictions for observations.

## How the new forecasts were generated

- **ZHVI:** Ridge, `alpha=100`, county fixed effects and Q2–Q4 quarter terms.
- **ZORI:** Elastic Net, `alpha=0.1`, `l1_ratio=0.75`, county fixed effects and no quarter terms.
- Both fit **one-year-ahead growth**. A future level equals the current level
  multiplied by `1 + predicted_growth_pct / 100`.
- The ten numeric predictors and preprocessing match the published
  [model comparison](checkpoint4_model_compare.md): housing quarterly/yearly
  momentum, mortgage rate and its yearly change, inflation, county unemployment
  and its yearly change, and price-to-rent ratio.
- The final fitting sample remains **216 county-quarters: 2015Q1–2019Q4 plus
  2021Q1–2021Q4**. Hyperparameters were previously chosen using 2021 validation.
  This extension does not tune again or fit to 2023–2024 test outcomes.
- Training medians and scaling are reused. **All ten predictors are present for
  all 36 forecast rows in 2025**; no 2025 predictor or future outcome is imputed.
  Missing 2025 ACS estimates are irrelevant to this particular selected feature set.

Before generating 2025 forecasts, the script independently reproduces all 72
saved test predictions for each selected model within `1e-8` percentage points.
Existing 2023–2024 forecasts and all historical observations remain unchanged.

## Limitations to read with the results

**2025Q4 has partial inputs.** County labor and inflation measures use
November–December observations. Flags are retained in the exported forecasts,
and the calculator displays a note for this quarter. See the
[integration notes](checkpoint1_data_integration.md) for the source rule.

The input panel is a retrospective project snapshot and may contain revisions
or values released after the labeled quarter. These forecasts are not an
as-of-release backtest, and they have not been evaluated against 2026 outcomes.
Model choices remain tentative for the reasons documented in the comparison
report. The calculator's adjustable sensitivity range is a scenario, **not a
statistical confidence interval**.

## Files and reproduction

Run from the repository root:

```bash
python src/build_calculator_forecasts.py
```

| File | Purpose |
|---|---|
| [Forecast CSV](../results/financial_calculator/forecasts_2025.csv) | 72 rows: nine counties × four origins × two housing targets |
| [Forecast manifest](../results/financial_calculator/forecast_manifest.json) | Input hashes, model settings, library versions, missingness and reproduction checks |
| [Calculator HTML](../financial_calculator.html) | Self-contained app; the script updates only its 2025 model-forecast values |

Forecast CSV keys are `county_fips`, `year`, `quarter`, and `target`.
`forecast_origin` is the decision quarter; `target_period` is four quarters later.
`predicted_growth_pct` is a percent, while `current_level` and
`predicted_level_t_plus_4` are dollars for ZHVI and dollars/month for ZORI.
The two `actual_*` columns stay empty. `outcome_status` explicitly records that
the future outcome is not in the project. The two `*_partial*` flags describe
the economic inputs, not missing target outcomes.

## README map and illustrations

Run `python src/build_readme_figures.py` to rebuild the figures from the checked-in
[Ohio county boundaries](../data/metadata/ohio_county_boundaries_2024.geojson).
The display-only GeoJSON contains Ohio's 88 counties, extracted without changing
coordinates from the [Census 2024 county KML, 1:20,000,000](https://www2.census.gov/geo/tiger/GENZ2024/kml/cb_2024_us_county_20m.zip).
Only the nine study counties are highlighted; the other counties are geographic
context and are not added to the modeling panel.

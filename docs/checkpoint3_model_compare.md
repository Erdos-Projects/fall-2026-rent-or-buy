# Checkpoint 3 Model Comparison

## What each model fits

All eight model families forecast a Zillow outcome four quarters after the prediction origin. The six pooled, non-time-series families are each evaluated with **two different fitting targets**. The first fits one-year growth and converts its prediction back to a future level; the second fits the future original level directly. ARIMA and ARIMAX fit the original level series. The mean baselines are named separately because their fitting targets differ.

| # | Model | Fitting target(s) | Forecast evaluated |
|---:|---|---|---|
| 1 | Mean growth baseline; Mean level baseline | One-year-ahead growth; original level at t+4 | Original level at t+4 |
| 2 | Ordinary Linear Regression | One-year-ahead growth **and** original level at t+4 | Original level at t+4 |
| 3 | LASSO | One-year-ahead growth **and** original level at t+4 | Original level at t+4 |
| 4 | Elastic Net | One-year-ahead growth **and** original level at t+4 | Original level at t+4 |
| 5 | Ridge | One-year-ahead growth **and** original level at t+4 | Original level at t+4 |
| 6 | XGBoost | One-year-ahead growth **and** original level at t+4 | Original level at t+4 |
| 7 | ARIMA | Original level time series; four-step forecast | Original level at t+4 |
| 8 | ARIMAX | Original level time series with exogenous predictors; four-step forecast | Original level at t+4 |

For a growth-fitted model, `predicted level(t+4) = observed level(t) × [1 + predicted growth(t→t+4) / 100]`. Direct-level models output `predicted level(t+4)` without that conversion. The realized direct-level label is reconstructed from the observed origin level and the supplied one-year growth label, so all methods are scored against the same outcome. The Ordinary Linear Regression row is the pooled, unregularized OLS model previously called “Pooled AR(1) dynamic panel”; its predictors include lagged housing growth and county fixed effects, but it is not a separately estimated AR(1) time-series model.

## Comparison design

The nine-county panel uses the project's existing chronological split: train origins **2015Q1–2019Q4** (180 county-quarters), validation origins **2021Q1–2021Q4** (36), and final test origins **2023Q1–2024Q4** (72). The intervening years are excluded as scored origins to separate one-year target windows. Pooled model hyperparameters and the choice between no quarter terms and Q2–Q4 indicators (Q1 reference) are selected on validation only. The primary selection metric is **validation RMSE on the original t+4 Zillow level**, with MAE as a tie-breaker. MAE and R² are shown as additional validation criteria; the test period is an out-of-period check, not a selection set. Pooled test fits use training plus validation observations after specifications are frozen.

Every predictor-based pooled model uses the same predeclared ten origin-quarter features: `zhvi_growth_qoq_pct`, `zhvi_growth_prev_1y_pct`, `zori_growth_qoq_pct`, `zori_growth_prev_1y_pct`, `nat_mortgage_rate_30y_pct`, `mortgage_rate_change_prev_1y_pp`, `nat_inflation_yoy_pct`, `cty_unemployment_rate_pct`, `cty_unemployment_change_prev_1y_pp`, and `price_to_rent_ratio`. These cover short and annual housing momentum, mortgage conditions and inflation, local labor conditions, and a housing affordability ratio. County fixed effects are included in each pooled fit. Direct-level predictor models additionally receive the **current level of the same Zillow series**; this level is also used to convert growth predictions to future levels, so both pathways are anchored to information known at the origin. Mean baselines omit numeric features but retain county effects. Median imputation and scaling are fit on the fitting partition only. The fixed feature list was selected by domain rationale before this expanded comparison, not by screening the test set; the validation selection still has uncertainty because it covers only four quarters.

ARIMA and ARIMAX forecast each county's level history separately to avoid treating county boundaries as continuous time. Their order, ARIMAX economic predictor bundle, and quarter terms are selected on validation. Future ARIMAX predictors are held at their origin values, so forecasts depend on that persistence assumption. The unchanged ARIMA/ARIMAX search and forecasts from PR #15 are reused by default to avoid thousands of redundant fits; the analysis script can recompute them with `--recompute-time-series`. Its cache check verifies that saved observed outcomes match the current split.

For Elastic Net, `l1_ratio=0.75` gives the L1 (LASSO) penalty a mixing weight of 0.75 and the L2 (Ridge) penalty a weight of 0.25. These are penalty weights, not literal shares of a predictor's influence or of total realized shrinkage.

## Validation selection

| Target | Metric | Validation winner | Fitting target | Quarter terms | Value |
|---|---|---|---|---|---:|
| ZHVI | RMSE | LASSO | Original level at t+4 | Yes | $5,180.00 |
| ZHVI | MAE | LASSO | Original level at t+4 | Yes | $3,908.33 |
| ZHVI | R² | LASSO | Original level at t+4 | Yes | 0.9933 |
| ZORI | RMSE | Elastic Net | One-year-ahead growth | No | $35.30/month |
| ZORI | MAE | Elastic Net | One-year-ahead growth | No | $29.99/month |
| ZORI | R² | Elastic Net | One-year-ahead growth | No | 0.9711 |

The **tentative selections under the declared validation rule** are direct-level LASSO for ZHVI and growth-fitted Elastic Net for ZORI. For ZHVI, this updates PR #15's tentative Ridge choice because direct-level models were absent from that comparison. The detailed validation results, selected seasonal versions, and metric winners are in [`validation_model_comparison.csv`](../results/checkpoint3_model_compare/validation_model_comparison.csv), [`validation_seasonality_comparison.csv`](../results/checkpoint3_model_compare/validation_seasonality_comparison.csv), and [`validation_metric_winners.csv`](../results/checkpoint3_model_compare/validation_metric_winners.csv).

![Validation RMSE by model](../results/checkpoint3_model_compare/validation_rmse_model_comparison.png)

*Figure 1. Validation root mean squared error (RMSE) on the **original level at t+4**: USD for ZHVI and USD/month for ZORI. Each x-axis label gives model, fitted target (growth or level), and selected quarter-term setting. Lower bars are better. LASSO fitted to levels leads ZHVI; Elastic Net fitted to growth leads ZORI. The large mean-level and direct-level XGBoost errors extend the vertical scale, so use the table above for close comparisons.*

![Validation MAE by model](../results/checkpoint3_model_compare/validation_mae_model_comparison.png)

*Figure 2. Validation mean absolute error (MAE) on the **same original t+4 level scale**. Lower is better. The x-axis follows the model order in the first table, with separate growth-fit and level-fit bars for each pooled family. Both targets retain the same winners as RMSE.*

![Validation R-squared by model](../results/checkpoint3_model_compare/validation_r2_model_comparison.png)

*Figure 3. Validation out-of-sample R² on the **original t+4 level scale**. Higher is better; zero is the mean-observed-level reference on this validation set, and negative bars are worse than that reference. Direct-level LASSO is highest for ZHVI; growth-fitted Elastic Net is highest for ZORI.*

## Interpreting the winning models

The following is **validation permutation importance**, not SHAP or causal attribution. Each feature is shuffled 30 times across validation county-quarters while the fitted model and other columns stay fixed. The bar reports the average increase in level-scale validation RMSE; a positive value means shuffling worsens the forecast. Correlated predictors can share or mask importance, and these rankings are unstable with only 36 validation rows. County and quarter indicators are controls and are not included in these plots.

![ZHVI LASSO feature importance](../results/checkpoint3_model_compare/zhvi_winning_model_feature_importance.png)

*Figure 4. ZHVI winner: direct-level LASSO. The current ZHVI level overwhelmingly dominates prediction. It uses a separate horizontal scale on the left; the right panel expands the smaller remaining effects. Among those, price-to-rent ratio and the prior-year mortgage-rate change have the largest positive permutation effects. The two panels have different x-axis ranges.*

![ZORI Elastic Net feature importance](../results/checkpoint3_model_compare/zori_winning_model_feature_importance.png)

*Figure 5. ZORI winner: growth-fitted Elastic Net. Prior-year ZHVI growth and prior-year mortgage-rate change have the largest positive permutation effects. Zero-height bars mean shuffling did not change the fitted predictions in this validation sample, often because regularization set their coefficients to zero. The x-axis is increase in original-level RMSE, in USD/month.*

The exact numerical importance values are in [`winning_model_feature_importance.csv`](../results/checkpoint3_model_compare/winning_model_feature_importance.csv). The dominance of current ZHVI for a direct-level fit is expected: it measures persistence of the level, not proof that the model explains future appreciation.

## Locked test check and R²

| Target | Validation-selected model | Test level MAE | Test level RMSE | Test level R² | Comparable growth-fit reference |
|---|---|---:|---:|---:|---|
| ZHVI | Direct-level LASSO | $14,538.32 | $18,072.77 | 0.9289 | Growth-fitted Ridge: $6,166.55 RMSE, 0.9917 R² |
| ZORI | Growth-fitted Elastic Net | $23.32/month | $28.89/month | 0.9841 | Direct-level ARIMAX: $16.10/month RMSE, 0.9951 R² |

The 2023–2024 test reveals a material regime shift. The validation-selected direct-level ZHVI LASSO generalizes poorly relative to growth-fitted Ridge; ZORI's validation-selected Elastic Net is also outperformed by ARIMAX on test. **We do not change the selected models using test results**, because that would turn the final test into another tuning set. These results weaken any claim that a single validation year reliably identifies a deployable winner. The full locked test table is in [`metrics.csv`](../results/checkpoint3_model_compare/metrics.csv); growth-scale diagnostics are in [`one_year_ahead_growth_test_metrics.csv`](../results/checkpoint3_model_compare/one_year_ahead_growth_test_metrics.csv).

Out-of-sample `R² = 1 − SSE/SST`, where `SSE` is the model's sum of squared forecast errors and `SST` is the sum of squared deviations of **observed outcomes in the scored split** from that split's observed mean. Thus a test level R² of `0.992` means model SSE is `0.008 × SST`, or about **99.2% lower** than the test-mean reference SSE. A test growth R² of `−0.634` means model SSE is `1.634 × SST`, or **63.4% higher** than that reference SSE. The test mean is used only to define the metric; it is not an available forecasting predictor. A high level R² can coexist with poor growth R² because stable differences between county price levels are easy to capture even when changing growth rates are not.

### Test time-series views

![ZHVI growth comparison](../results/checkpoint3_model_compare/zhvi_1y_ahead_growth_test_prediction_comparison.png)

*Figure 6. The y-axis is **average one-year-ahead ZHVI growth (%)** across nine counties at each 2023–2024 prediction origin: `100 × [ZHVI(t+4)/ZHVI(t) − 1]`. Black is observed growth; colored curves are model forecasts converted to growth where needed. Separate panels distinguish each fitted target. Shading gives approximate 95% bands; pooled bands are quarter-block bootstrap refit bands, whereas ARIMA/ARIMAX bands are conditional forecast intervals.*

![ZORI growth comparison](../results/checkpoint3_model_compare/zori_1y_ahead_growth_test_prediction_comparison.png)

*Figure 7. The y-axis is **average one-year-ahead ZORI growth (%)** across the nine counties, defined analogously to Figure 6. The x-axis is the prediction-origin quarter, not the future outcome quarter. Curves are county-average forecasts; interval methods differ across model families and should not be compared as though identically calibrated.*

![ZHVI level comparison](../results/checkpoint3_model_compare/zhvi_test_prediction_comparison.png)

*Figure 8. The y-axis is **average original ZHVI at t+4 (USD)** across nine counties. Growth-fitted forecasts are converted to levels using observed ZHVI at the origin; direct-level and time-series forecasts already output levels. Black is observed outcome, colors are forecasts, shading is the model-specific 95% band.*

![ZORI level comparison](../results/checkpoint3_model_compare/zori_test_prediction_comparison.png)

*Figure 9. The y-axis is **average original ZORI at t+4 (USD/month)** across nine counties. The x-axis remains the 2023–2024 prediction-origin quarter. Growth and direct-level fitting targets appear in separate panels to make their outputs comparable without confusing what each model was trained to predict.*

## Limits and reproduction

The validation period contains just four quarters and nine counties; the post-2022 test has a different housing and interest-rate environment. The new direct-level LASSO result is especially sensitive to this shift. The test should remain untouched for any further model changes; a future project phase could add more historical validation folds or a prospective holdout. Pooled bootstrap bands reflect refit variation, while ARIMA/ARIMAX intervals are conditional on their chosen specification and, for ARIMAX, fixed future exogenous values. Neither is a guarantee of empirical 95% coverage.

Run [`notebooks/pooled_model_comparison_zhvi_1y.ipynb`](../notebooks/pooled_model_comparison_zhvi_1y.ipynb) or [`src/model_comparison_analysis.py`](../src/model_comparison_analysis.py) from the repository root. The script reads the existing one-year chronological splits and feature panel and writes comparison tables and figures to [`results/checkpoint3_model_compare/`](../results/checkpoint3_model_compare/). Use `python src/model_comparison_analysis.py --recompute-time-series` to refit the slower ARIMA/ARIMAX search rather than reusing the checked-in, unchanged PR #15 time-series results. Person 3's distinct time-series work remains in [`results/checkpoint3_time_series/`](../results/checkpoint3_time_series/).

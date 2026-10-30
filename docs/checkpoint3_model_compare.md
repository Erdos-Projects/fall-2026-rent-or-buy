# Checkpoint 3 Model Comparison

## Research target and model-selection rule

The project's primary prediction target is **one-year-ahead ZHVI and ZORI growth**, measured as `100 × [level(t+4) / level(t) − 1]`. **Eligible candidates** are pooled models fitted directly to that growth outcome **plus ARIMA and ARIMAX**. The time-series models forecast each county's original level, then imply one-year growth with the same equation using the observed level at the prediction origin. Pooled regressions fitted directly to future levels remain **reference comparisons only**. This is an explicit model-family policy, not a claim that direct-level fitting is inherently worse. Among eligible candidates, we select the specification with the lowest validation RMSE on the original t+4 level, with MAE as a tie-breaker and R² as a supporting metric. Level-scale scoring supports the downstream rent-versus-buy calculator's need for future dollar values; it does not change the project's growth outcome.

## What each model fits

All eight model families forecast a Zillow outcome four quarters after the prediction origin. The six pooled, non-time-series families are each evaluated with **two different fitting targets**. The first fits one-year growth and converts its prediction back to a future level; the second fits the future original level directly. ARIMA and ARIMAX fit the original level series. The mean baselines are named separately because their fitting targets differ.

| # | Model | Fitting target(s) | Forecast evaluated | Selection eligibility |
|---:|---|---|---|---|
| 1 | Mean growth baseline; Mean level baseline | One-year-ahead growth; original level at t+4 | Original level at t+4 | Growth baseline eligible; level baseline reference only |
| 2 | Ordinary Linear Regression | One-year-ahead growth **and** original level at t+4 | Original level at t+4 | Growth fit eligible; direct-level fit reference only |
| 3 | LASSO | One-year-ahead growth **and** original level at t+4 | Original level at t+4 | Growth fit eligible; direct-level fit reference only |
| 4 | Elastic Net | One-year-ahead growth **and** original level at t+4 | Original level at t+4 | Growth fit eligible; direct-level fit reference only |
| 5 | Ridge | One-year-ahead growth **and** original level at t+4 | Original level at t+4 | Growth fit eligible; direct-level fit reference only |
| 6 | XGBoost | One-year-ahead growth **and** original level at t+4 | Original level at t+4 | Growth fit eligible; direct-level fit reference only |
| 7 | ARIMA | Original level time series; four-step forecast | Original level at t+4 | Eligible time-series candidate |
| 8 | ARIMAX | Original level time series with exogenous predictors; four-step forecast | Original level at t+4 | Eligible time-series candidate |

For a growth-fitted model, `predicted level(t+4) = observed level(t) × [1 + predicted growth(t→t+4) / 100]`. Direct-level models output `predicted level(t+4)` without that conversion. The realized direct-level label is reconstructed from the observed origin level and the supplied one-year growth label, so all methods are scored against the same outcome. The Ordinary Linear Regression row is the pooled, unregularized OLS model previously called “Pooled AR(1) dynamic panel”; its predictors include lagged housing growth and county fixed effects, but it is not a separately estimated AR(1) time-series model.

## Comparison design

The nine-county panel uses the project's existing chronological split: train origins **2015Q1–2019Q4** (180 county-quarters), validation origins **2021Q1–2021Q4** (36), and test origins **2023Q1–2024Q4** (72). The intervening years are excluded as scored origins to separate one-year target windows. Pooled model hyperparameters and the choice between no quarter terms and Q2–Q4 indicators (Q1 reference) are selected on validation only. The candidate-eligibility and level-scale validation scoring rules are stated above. Pooled test fits use training plus validation observations after specifications are frozen. **Although growth was the project's original research outcome, this explicit candidate-pool policy was formalized after the team had seen test results from the earlier comparison; the existing test is therefore descriptive and is no longer a pristine holdout for evaluating the revised policy.**

Every predictor-based pooled model uses the same predeclared ten origin-quarter features: `zhvi_growth_qoq_pct`, `zhvi_growth_prev_1y_pct`, `zori_growth_qoq_pct`, `zori_growth_prev_1y_pct`, `nat_mortgage_rate_30y_pct`, `mortgage_rate_change_prev_1y_pp`, `nat_inflation_yoy_pct`, `cty_unemployment_rate_pct`, `cty_unemployment_change_prev_1y_pp`, and `price_to_rent_ratio`. These cover short and annual housing momentum, mortgage conditions and inflation, local labor conditions, and a housing affordability ratio. County fixed effects are included in each pooled fit. Direct-level predictor models additionally receive the **current level of the same Zillow series**; this level is also used to convert growth predictions to future levels, so both pathways are anchored to information known at the origin. Mean baselines omit numeric features but retain county effects. Median imputation and scaling are fit on the fitting partition only. The fixed feature list was selected by domain rationale before this expanded comparison, not by screening the test set; the validation selection still has uncertainty because it covers only four quarters.

ARIMA and ARIMAX forecast each county's level history separately to avoid treating county boundaries as continuous time. Their order, ARIMAX economic predictor bundle, and quarter terms are selected on validation. Future ARIMAX predictors are held at their origin values, so forecasts depend on that persistence assumption. The unchanged ARIMA/ARIMAX search and forecasts from PR #15 are reused by default to avoid thousands of redundant fits; the analysis script can recompute them with `--recompute-time-series`. Its cache check verifies that saved observed outcomes match the current split.

For Elastic Net, `l1_ratio=0.75` gives the L1 (LASSO) penalty a mixing weight of 0.75 and the L2 (Ridge) penalty a weight of 0.25. These are penalty weights, not literal shares of a predictor's influence or of total realized shrinkage.

### Does fitting levels work worse than fitting growth?

Not uniformly on validation. Comparing each of the five predictor-based pooled families with itself, the growth-fitted version has lower validation level RMSE in **2/5 ZHVI** and **5/5 ZORI** comparisons. For ZHVI, direct-level LASSO reaches $5,180 validation RMSE, below growth-fitted Ridge's $5,989. Growth fits have lower **test** RMSE in 5/5 families for each target, but that observation cannot serve as an independent justification for a post-test restriction. Direct-level regressions remain references under the stated candidate-pool policy; ARIMA and ARIMAX remain eligible time-series alternatives even though they forecast levels first. The paired validation values are saved in [`fit_target_paired_validation_comparison.csv`](../results/checkpoint3_model_compare/fit_target_paired_validation_comparison.csv).

## Validation selection

| Target | Tentative model | Quarter terms | Validation level MAE | Validation RMSE | Validation R² | Why |
|---|---|---|---:|---:|---:|---|
| ZHVI | Ridge (growth fit; alpha=100) | Yes | $4,464.04 | $5,988.60 | 0.9910 | Lowest validation level RMSE among eligible growth-fit and ARIMA/ARIMAX candidates. |
| ZORI | Elastic Net (growth fit; alpha=0.1, l1_ratio=0.75) | No | $29.99/month | $35.30/month | 0.9711 | Lowest validation level RMSE among eligible growth-fit and ARIMA/ARIMAX candidates. |

For each target, the same eligible model also has the lowest validation MAE and highest R². The tentative choices remain **ZHVI Ridge and ZORI Elastic Net** when ARIMA/ARIMAX are included. The unrestricted all-candidate winner is different for ZHVI: direct-level LASSO. The detailed validation results, each row's eligibility flag, and metric winners are in [`validation_model_comparison.csv`](../results/checkpoint3_model_compare/validation_model_comparison.csv), [`validation_seasonality_comparison.csv`](../results/checkpoint3_model_compare/validation_seasonality_comparison.csv), [`validation_metric_winners.csv`](../results/checkpoint3_model_compare/validation_metric_winners.csv), and [`validation_metric_winners_all_candidates.csv`](../results/checkpoint3_model_compare/validation_metric_winners_all_candidates.csv).

![Validation RMSE by model](../results/checkpoint3_model_compare/validation_rmse_model_comparison.png)

*Figure 1. Top: ZHVI validation RMSE in USD; bottom: ZORI validation RMSE in USD/month. Both score the **original level at t+4**; growth-fitted predictions are converted to that level. Each bar has its numerical value above it. The orange bar with a bold **\*** marks the winner among **eligible growth-fit and ARIMA/ARIMAX** candidates: ZHVI Ridge and ZORI Elastic Net. Blue bars are growth fits; slate bars are eligible ARIMA/ARIMAX forecasts; pale gray bars are reference-only direct-level regressions. Direct-level ZHVI LASSO has a smaller unrestricted RMSE than starred Ridge but is ineligible under the candidate-pool rule.*

![Validation MAE by model](../results/checkpoint3_model_compare/validation_mae_model_comparison.png)

*Figure 2. Top: ZHVI validation MAE in USD; bottom: ZORI validation MAE in USD/month, both on the **original t+4 level scale**. Lower is better. Values appear above each bar; the orange bar and bold **\*** identify the eligible MAE winner across growth fits and ARIMA/ARIMAX. Model labels also give fitting target and quarter-term choice. Direct-level regression results remain visible for comparison.*

![Validation R-squared by model](../results/checkpoint3_model_compare/validation_r2_model_comparison.png)

*Figure 3. Top: ZHVI validation R²; bottom: ZORI validation R², both on the **original t+4 level scale**. Higher is better; zero is the validation-set observed-mean reference. Values appear at each bar, and the orange bar with bold **\*** identifies the eligible R² winner across growth fits and ARIMA/ARIMAX. Direct-level ZHVI LASSO has the highest unrestricted R² but is a reference comparison.*

## Interpreting the winning models

The charts below show **permutation sensitivity** for the two growth-fitted tentative models, not SHAP values, causal effects, or significance tests. A national feature is shuffled as a whole quarter block so it remains identical across counties in that quarter; a county-level feature is shuffled across counties **within** each quarter. This avoids creating impossible county-specific national rates. Each feature is shuffled 30 times while the fitted model and other columns stay fixed. Bars show the mean percentage change in validation **level-scale** RMSE relative to the unshuffled model; whiskers show the standard deviation across random shuffles, **not a confidence interval**. Positive values mean shuffling worsens validation forecasts; negative values mean the shuffled values happened to improve RMSE. Correlated predictors may share or mask importance, and rankings from just 36 validation rows can be unstable. County and quarter indicators are held fixed and omitted from these charts.

![ZHVI Ridge permutation sensitivity](../results/checkpoint3_model_compare/zhvi_winning_model_feature_importance.png)

*Figure 4. ZHVI tentative model: growth-fitted Ridge. The horizontal axis is percentage change in validation original-level RMSE after panel-preserving permutation; darker blue marks the three largest positive sensitivity values. County unemployment change (+7.4%), county unemployment rate (+5.7%), and mortgage-rate change (+4.0%) rank highest here. The plot excludes current ZHVI level because Ridge fits growth, then uses the observed origin level only to convert its growth forecast to a future level. Read bars as predictive sensitivity on this validation sample, not coefficient direction or causal influence.*

![ZORI Elastic Net feature importance](../results/checkpoint3_model_compare/zori_winning_model_feature_importance.png)

*Figure 5. ZORI tentative model: growth-fitted Elastic Net. The horizontal axis is percentage change in validation original-level RMSE after panel-preserving permutation. Mortgage-rate change (+6.4%) and county unemployment rate (+2.4%) show the largest positive sensitivity. Shuffling inflation improves RMSE (−6.6%), a warning that its estimated contribution may not generalize; this is not evidence of a protective causal effect. Near-zero bars can arise from small or zero fitted coefficients, redundancy among correlated inputs, or limited validation variation; they do not prove a feature is irrelevant.*

The exact numerical sensitivity values and permutation rules are in [`winning_model_feature_importance.csv`](../results/checkpoint3_model_compare/winning_model_feature_importance.csv). A next-step robustness check would repeat these plots over multiple forward validation periods; the current four-quarter validation year cannot establish stable feature rankings.

## Locked test check and R²

| Target | **Validation-selected model** | Quarter terms | Test level MAE | Test level RMSE | Test level R² |
|---|---|---|---:|---:|---:|
| ZHVI | Ridge (growth fit) | Yes | $4,736.47 | $6,166.55 | 0.9917 |
| ZORI | Elastic Net (growth fit) | No | $23.32/month | $28.89/month | 0.9841 |

The 2023–2024 test remains a difficult distribution-shift check. Direct-level ZHVI LASSO, the unrestricted validation winner, has $18,072.77 test RMSE versus $6,166.55 for the growth-fitted Ridge choice. ZORI Elastic Net is outperformed on test by **eligible ARIMAX** ($16.10/month RMSE), but ARIMAX had higher validation RMSE ($62.50/month versus Elastic Net's $35.30/month); the test result does not change the validation-selected choice. The model parameters were not retuned on the test, yet the candidate-pool policy was formalized **after these test outcomes were known**. Accordingly, these scores are descriptive comparisons, not an unbiased final assessment of that revised policy. The full test table is in [`metrics.csv`](../results/checkpoint3_model_compare/metrics.csv); growth-scale diagnostics are in [`one_year_ahead_growth_test_metrics.csv`](../results/checkpoint3_model_compare/one_year_ahead_growth_test_metrics.csv).

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

The validation period contains just four quarters and nine counties; the post-2022 test has a different housing and interest-rate environment. Direct-level LASSO's strong validation result but weak test result illustrates this instability. Because the team has already inspected the 2023–2024 results, a future unbiased performance assessment needs new prospective data or a separately designed forward validation period. Pooled bootstrap bands reflect refit variation, while ARIMA/ARIMAX intervals are conditional on their chosen specification and, for ARIMAX, fixed future exogenous values. Neither is a guarantee of empirical 95% coverage.

Run [`notebooks/pooled_model_comparison_zhvi_1y.ipynb`](../notebooks/pooled_model_comparison_zhvi_1y.ipynb) or [`src/model_comparison_analysis.py`](../src/model_comparison_analysis.py) from the repository root. The script reads the existing one-year chronological splits and feature panel and writes comparison tables and figures to [`results/checkpoint3_model_compare/`](../results/checkpoint3_model_compare/). Use `python src/model_comparison_analysis.py --recompute-time-series` to refit the slower ARIMA/ARIMAX search rather than reusing the checked-in, unchanged PR #15 time-series results. Person 3's distinct time-series work remains in [`results/checkpoint3_time_series/`](../results/checkpoint3_time_series/).

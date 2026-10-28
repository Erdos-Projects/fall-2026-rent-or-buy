# Checkpoint 3 Model Comparison

## What each model predicts

| # | Model | Model-fitting target | Forecast shown in the main comparison |
|---:|---|---|---|
| 1 | Mean baseline | One-year-ahead ZHVI/ZORI growth | Predict growth, then convert it to the Zillow level at t+4 |
| 2 | LASSO | One-year-ahead ZHVI/ZORI growth | Predict growth, then convert it to the Zillow level at t+4 |
| 3 | Elastic Net | One-year-ahead ZHVI/ZORI growth | Predict growth, then convert it to the Zillow level at t+4 |
| 4 | Ridge | One-year-ahead ZHVI/ZORI growth | Predict growth, then convert it to the Zillow level at t+4 |
| 5 | XGBoost | One-year-ahead ZHVI/ZORI growth | Predict growth, then convert it to the Zillow level at t+4 |
| 6 | Pooled AR(1) dynamic panel | One-year-ahead ZHVI/ZORI growth | Predict growth, then convert it to the Zillow level at t+4 |
| 7 | ARIMAX | Original ZHVI/ZORI level at t+4 | Forecast the level directly |
| 8 | ARIMA | Original ZHVI/ZORI level at t+4 | Forecast the level directly |

Figures 1–2 show one-year-ahead growth. Figures 3–4 show the common future-level scale. For models 1–6: `predicted level(t+4) = level(t) × (1 + predicted growth(t+4) / 100)`. ARIMA and ARIMAX forecast the level directly.

## Goal and evaluation design

We compared pooled regression and time-series approaches for one-year-ahead housing outcomes in the nine Ohio counties. For the pooled growth models, the target is the engineered one-year-ahead percentage growth (`zhvi_growth_ahead_1y_pct` or `zori_growth_ahead_1y_pct`). Predicted growth is converted to the Zillow value one year later using the observed origin-quarter value:

`predicted level(t+4) = level(t) × (1 + predicted growth(t+4) / 100)`.

ARIMA and ARIMAX instead forecast the original Zillow level. This lets every model be evaluated on the same future ZHVI or ZORI level. Quarterly rows are Q1–Q4, and the common chronological split is train 2015Q1–2019Q4, validation 2021Q1–2021Q4, and test 2023Q1–2024Q4. The gap years 2020 and 2022 are excluded from scored origins to prevent target-window overlap/leakage and honor the selected split.

The comparison includes a mean baseline, LASSO, Elastic Net, Ridge, XGBoost, pooled AR(1) dynamic panel, ARIMAX, and ARIMA. The five predictor-based pooled models (LASSO, Elastic Net, Ridge, XGBoost, and pooled AR(1)) share this predeclared ten-variable set: `zhvi_growth_qoq_pct`, `zhvi_growth_prev_1y_pct`, `zori_growth_qoq_pct`, `zori_growth_prev_1y_pct`, `nat_mortgage_rate_30y_pct`, `mortgage_rate_change_prev_1y_pp`, `nat_inflation_yoy_pct`, `cty_unemployment_rate_pct`, `cty_unemployment_change_prev_1y_pp`, and `price_to_rent_ratio`. The mean baseline is predictor-free. All pooled models use county fixed effects, with Q1 as the reference when quarter indicators are included. ARIMA/ARIMAX use each county's own history (rather than concatenating counties into a false continuous series); ARIMAX order and a limited economic predictor bundle were selected on validation. Future ARIMAX macro predictors are held at their last observed values, so those forecasts are conditional on a persistence assumption.

For each target/model, versions with and without Q2–Q4 indicators were compared on validation. The model/seasonality selection criterion was **validation RMSE on original Zillow levels**, with MAE as tie-breaker; test data were not used for selection. Reported 95% uncertainty bands are quarter-block bootstrap refit bands for pooled regression models and conditional forecast intervals for ARIMA/ARIMAX. These intervals are not directly equivalent: the time-series intervals condition on the fitted specification, and ARIMAX bands omit uncertainty in future exogenous variables.

## Tentative model choices

| Target | Tentative model | Quarter terms | Validation level RMSE | Validation MAE | Validation R² | Why |
|---|---|---:|---:|---:|---:|---|
| ZHVI | Ridge (alpha=100) | Yes | $5,989 | $4,464 | 0.991 | Lowest validation level RMSE among evaluated models for ZHVI. |
| ZORI | Elastic Net (alpha=0.1, l1_ratio=0.25) | No | $35.30/month | $29.99/month | 0.971 | Lowest validation level RMSE among evaluated models for ZORI. |

These are **tentative prediction choices**, not a claim that either model fully captures the market regime change after 2022. They were selected by the validation criterion above; the 2023–2024 test is retained as an out-of-period assessment and was not used to choose models.

## Test performance on the original level scale

The table reports the locked test-set level metrics for all eight approaches. Lower MAE/RMSE is better; R² is relative to the test-set mean and should be interpreted cautiously because a strong common time trend can make R² look high even when a model misses quarter-to-quarter changes.

| Target | Model | Quarter terms | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|---:|
| ZHVI | ARIMA | Yes | $9,200.31 | $11,546.58 | 0.971 |
| ZHVI | ARIMAX | No | $8,088.64 | $10,192.02 | 0.977 |
| ZHVI | Elastic Net | Yes | $5,578.63 | $7,279.88 | 0.988 |
| ZHVI | LASSO | Yes | $6,487.17 | $8,429.19 | 0.985 |
| ZHVI | Mean baseline | No | $6,831.66 | $9,154.57 | 0.982 |
| ZHVI | Pooled AR(1) dynamic panel | Yes | $8,013.93 | $10,660.80 | 0.975 |
| ZHVI | Ridge | Yes | $4,736.47 | $6,166.55 | 0.992 |
| ZHVI | XGBoost | Yes | $6,357.47 | $8,377.98 | 0.985 |
| ZORI | ARIMA | No | $43.17/month | $45.89/month | 0.960 |
| ZORI | ARIMAX | No | $12.83/month | $16.10/month | 0.995 |
| ZORI | Elastic Net | No | $23.32/month | $28.89/month | 0.984 |
| ZORI | LASSO | No | $23.87/month | $29.23/month | 0.984 |
| ZORI | Mean baseline | No | $14.62/month | $18.67/month | 0.993 |
| ZORI | Pooled AR(1) dynamic panel | No | $21.76/month | $26.93/month | 0.986 |
| ZORI | Ridge | No | $25.21/month | $31.39/month | 0.981 |
| ZORI | XGBoost | Yes | $14.29/month | $17.75/month | 0.994 |

## Test performance on the one-year growth scale

Because the primary prediction target is one-year-ahead growth, we also convert each predicted future level back into an implied growth rate and report growth metrics. These are **diagnostic test metrics**: model tuning and quarter-term selection above minimized validation level RMSE, not growth-rate RMSE. Negative growth-scale R² for several models shows that the chosen level forecasts can still miss the direction and size of growth changes, especially across a market regime shift.

| Target | Model | Quarter terms | MAE (percentage points) | RMSE (percentage points) | R² |
|---|---|---:|---:|---:|---:|
| ZHVI | Ridge | Yes | 2.002 | 2.521 | -0.634 |
| ZHVI | Elastic Net | Yes | 2.340 | 2.920 | -1.192 |
| ZHVI | XGBoost | Yes | 2.453 | 2.969 | -1.267 |
| ZHVI | LASSO | Yes | 2.659 | 3.242 | -1.703 |
| ZHVI | Mean baseline | No | 2.611 | 3.253 | -1.722 |
| ZHVI | ARIMAX | No | 3.149 | 3.706 | -2.532 |
| ZHVI | Pooled AR(1) dynamic panel | Yes | 3.234 | 3.987 | -3.088 |
| ZHVI | ARIMA | Yes | 3.591 | 4.241 | -3.625 |
| ZORI | ARIMAX | No | 1.003 | 1.246 | -0.328 |
| ZORI | XGBoost | Yes | 1.097 | 1.326 | -0.504 |
| ZORI | Mean baseline | No | 1.127 | 1.401 | -0.680 |
| ZORI | Pooled AR(1) dynamic panel | No | 1.748 | 2.211 | -3.182 |
| ZORI | Elastic Net | No | 1.856 | 2.336 | -3.670 |
| ZORI | LASSO | No | 1.904 | 2.368 | -3.799 |
| ZORI | Ridge | No | 1.997 | 2.489 | -4.303 |
| ZORI | ARIMA | No | 3.422 | 3.637 | -10.323 |

## Figures

The figures below compare the nine-county average actual test series with model-average forecasts. In each panel, the black line is the observed series, the colored line is the forecast, and the shaded region is the model's 95% uncertainty band. The horizontal axis shows the prediction-origin quarter; the vertical axis shows either growth in percent or the predicted future Zillow level, as labeled. Narrower bands indicate less estimated uncertainty under that model's interval method, not guaranteed accuracy.

### ZHVI: one-year-ahead target growth

![ZHVI one-year-ahead growth predictions on the test period](../results/checkpoint3_model_compare/zhvi_1y_ahead_growth_test_prediction_comparison.png)

*Figure 1. The y-axis is average one-year-ahead ZHVI growth across the nine counties, in percent: 100 × (ZHVI at t+4 quarters / ZHVI at t − 1). Black shows observed growth; color shows model predictions; shading shows 95% uncertainty intervals. Panel subtitles identify each selected specification, predictors where applicable, and quarter terms.*

### ZORI: one-year-ahead target growth

![ZORI one-year-ahead growth predictions on the test period](../results/checkpoint3_model_compare/zori_1y_ahead_growth_test_prediction_comparison.png)

*Figure 2. The y-axis is average one-year-ahead ZORI growth across the nine counties, in percent: 100 × (ZORI at t+4 quarters / ZORI at t − 1). Black shows observed growth; color shows model predictions; shading shows 95% uncertainty intervals. Panel subtitles identify each selected specification, predictors where applicable, and quarter terms.*

### Model-specific level forecasts

![ZHVI test-period level forecasts](../results/checkpoint3_model_compare/zhvi_test_prediction_comparison.png)

*Figure 3. The y-axis is average future ZHVI across the nine counties, in U.S. dollars. Models 1–6 predict one-year growth and convert it to the future level; ARIMAX and ARIMA forecast levels directly. Shading marks 95% uncertainty intervals. Panel subtitles give selected model orders/hyperparameters, predictor bundle where applicable, and quarter terms.*

![ZORI test-period level forecasts](../results/checkpoint3_model_compare/zori_test_prediction_comparison.png)

*Figure 4. The y-axis is average future ZORI across the nine counties, in U.S. dollars per month. Models 1–6 predict one-year growth and convert it to the future level; ARIMAX and ARIMA forecast levels directly. Shading marks 95% uncertainty intervals. Panel subtitles give selected model orders/hyperparameters, predictor bundle where applicable, and quarter terms.*

## Appendix: separate direct-level experiment

Figures 5–6 come from a separate preliminary experiment, not the primary growth-to-level comparison in Figures 1–4. In this appendix experiment, **all eight models were fit directly to future levels**. Each Figure 5 panel predicts `zhvi_usd_ahead_1y`; each Figure 6 panel predicts `zori_usd_month_ahead_1y`. Thus the six non-time-series models here also fit levels, not growth rates. This different setup explains why the curves and rankings can differ from Figures 3–4. Treat these plots as exploratory; they are not used for primary model selection.

| Model shown in Figures 5–6 | Target fitted in this separate experiment |
|---|---|
| Mean baseline | Direct one-year-ahead ZHVI or ZORI level |
| LASSO regression | Direct one-year-ahead ZHVI or ZORI level |
| Elastic Net regression | Direct one-year-ahead ZHVI or ZORI level |
| Ridge regression | Direct one-year-ahead ZHVI or ZORI level |
| XGBoost | Direct one-year-ahead ZHVI or ZORI level |
| Pooled AR(1) dynamic panel | Direct one-year-ahead ZHVI or ZORI level |
| ARIMAX | Direct one-year-ahead ZHVI or ZORI level |
| ARIMA | Direct one-year-ahead ZHVI or ZORI level |

![Separate direct-level ZHVI forecasts](../results/checkpoint3_model_compare/zhvi/test_model_specific_predictions.png)

*Figure 5. Every model directly predicts one-year-ahead ZHVI level. The y-axis is average future ZHVI across nine counties, in U.S. dollars; black is observed, color is predicted, and shading shows 95% intervals. This is the separate direct-level experiment above.*

![Separate direct-level ZORI forecasts](../results/checkpoint3_model_compare/zori/test_model_specific_predictions.png)

*Figure 6. Every model directly predicts one-year-ahead ZORI level. The y-axis is average future ZORI across nine counties, in U.S. dollars per month; black is observed, color is predicted, and shading shows 95% intervals. This is the separate direct-level experiment above.*


## Limitations and interpretation

- There are only nine counties and a short pre-pandemic training window. The 2023–2024 test period follows a different rate and housing-market environment, so it is a meaningful but difficult distribution-shift test.
- The one-year validation period is small. Treat model ordering and quarter-term selection as provisional, and report test results separately rather than selecting on them.
- Ridge and Elastic Net are the tentative choices because they had the best validation RMSE on the level scale for their respective targets. Their growth-rate errors can still be poor; the conclusion should be framed as “selected under the stated validation criterion,” not “best at predicting the 2023–2024 slowdown.”
- Growth and level objectives differ. If the project prioritizes predicting appreciation/deceleration rather than future dollar levels, the team should predeclare growth-rate RMSE as the primary criterion and rerun model selection using validation growth metrics before finalizing the model.
- The bands use different uncertainty procedures across model families and are not directly comparable as calibrated coverage statements. The held-out test period is only eight quarters, so empirical coverage is imprecise.

## Reproducibility and artifacts

Run [`notebooks/pooled_model_comparison_zhvi_1y.ipynb`](../notebooks/pooled_model_comparison_zhvi_1y.ipynb) from the project directory. Input data and predictor definitions are loaded from `processed/` and `metadata/`. The corresponding metric tables, prediction series, interval tables, and plots are stored in [`results/checkpoint3_model_compare/`](../results/checkpoint3_model_compare/). Person 3's original growth-target time-series results remain in [`results/checkpoint3_time_series/`](../results/checkpoint3_time_series/). The early Person 4 time-series experiment was removed from the results folder.

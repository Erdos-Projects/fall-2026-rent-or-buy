# Checkpoint 3 Evaluation Plan

## 1. Evaluation Goal

Checkpoint 3 evaluates how well our models predict two housing-market targets:

- One-year-ahead ZHVI growth
- One-year-ahead ZORI growth

We use county-quarter data for nine Ohio counties. We removed the three-year targets because the available 2015–2025 data cannot support separate training, validation, and testing periods without overlapping future outcomes. The one-year predictions will eventually support the rent-or-buy financial calculator.

## 2. Chronological Data Split

We use a chronological split instead of randomly dividing the data. This lets the models learn from earlier quarters and predict later ones.

| Dataset | Prediction quarters | County-quarter rows |
|---|---|---:|
| Training | 2015 Q1–2019 Q4 | 180 |
| Validation | 2021 Q1–2021 Q4 | 36 |
| Testing | 2023 Q1–2024 Q4 | 72 |

The gaps prevent future target information from entering the training data. We use the validation data to select and tune the model. We keep the testing data untouched until the final evaluation. We exclude 2020 because annual ACS data are unavailable for that year, and we exclude 2022 because its one-year-ahead outcomes overlap with the testing period.

## 3. Models and Baseline

We compared eight approaches:

- Mean baseline
- Ordinary Linear Regression
- Ridge Regression
- LASSO Regression
- Elastic Net Regression
- XGBoost
- ARIMA
- ARIMAX

The mean baseline predicts the average one-year growth from the training data. Ordinary Linear Regression, Ridge, LASSO, Elastic Net, and XGBoost predict one-year growth using the selected housing and economic variables. ARIMA and ARIMAX forecast the future Zillow level directly. We used validation RMSE on the future Zillow level to select the models, with MAE used to break a tie.

## 4. Evaluation Metrics

We report three metrics:

- **Mean Absolute Error (MAE):** the average size of the prediction errors. Lower values are better.
- **Root Mean Squared Error (RMSE):** gives more weight to large errors. Lower values are better.
- **R²:** compares the model with an average prediction. Higher values are better. A negative R² means that the model performed worse than the average prediction.

The main comparison uses future Zillow levels so that all models can be evaluated on the same scale. We also report growth-scale results to show how well the models predict changes in home values and rents.


## 5. Preventing Data Leakage

We followed these rules to prevent the models from using future information:

- Used chronological periods with gap years in 2020 and 2022.
- Used only information available at the prediction quarter.
- Learned missing-value rules and scaling from the training data only.
- Used validation data for model selection and kept testing data untouched until the final evaluation.

## 6. Comparisons and Stress Tests

- Compared all eight models using the same chronological split.
- Compared ZHVI and ZORI results on both the future-level and growth scales.
- Compared models with and without quarterly indicators using validation data.
- A future stress test could remove pandemic-period observations and compare the results.

## 7. Results

Validation results selected Ridge Regression for ZHVI and Elastic Net for ZORI. We made these choices before examining the final testing results.

| Target | Selected model | Level MAE | Level RMSE | Level R² | Growth MAE | Growth RMSE | Growth R² |
|---|---|---:|---:|---:|---:|---:|---:|
| ZHVI | Ridge | $4,736.47 | $6,166.55 | 0.992 | 2.002 | 2.521 | -0.634 |
| ZORI | Elastic Net | $23.32/month | $28.89/month | 0.984 | 1.856 | 2.336 | -3.670 |

Both selected models had strong results when predicting future Zillow levels. However, their negative growth-scale R² values show that they did not predict changes in growth better than an average-growth prediction. The models had difficulty capturing the housing-market slowdown after 2022. Some other models performed better on individual testing measures, especially for ZORI. However, we did not change the selected models after seeing the testing results because the testing data must remain separate from model selection.

## 8. Limitations

- The dataset contains only nine Ohio counties.
- The training period is short and mostly comes before the pandemic.
- The 2023–2024 testing period had different interest-rate and housing-market conditions.
- The validation period contains only four quarters, and the testing period contains only eight quarters.
- Results on the future-level scale can differ from results on the growth scale.
- The uncertainty intervals use different methods across the model types, so they cannot be compared directly.

These limitations mean that the selected models and conclusions should be treated as preliminary.

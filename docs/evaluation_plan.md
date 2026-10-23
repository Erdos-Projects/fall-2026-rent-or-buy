# Checkpoint 3 Evaluation Plan

## 1. Evaluation Goal

Checkpoint 3 will evaluate how well our models predict four housing-market targets:

- One-year-ahead ZHVI growth
- One-year-ahead ZORI growth
- Three-year-ahead ZHVI growth
- Three-year-ahead ZORI growth

We use county-quarter data for nine Ohio counties. The models will use information from earlier quarters to predict later quarters. These predictions will eventually support the rent-or-buy financial calculator.

## 2. Chronological Data Splits

We will use chronological splits instead of randomly dividing the data. This allows the models to learn from earlier quarters and predict later quarters.

| Forecast horizon | Training quarters | Validation quarters | Testing quarters | Rows: train / validation / test |
|---|---|---|---|---|
| One year | 2015 Q1–2019 Q4 | 2021 Q1–2022 Q4 | 2023 Q1–2024 Q4 | 180 / 72 / 72 |
| Three years | 2015 Q1–2017 Q4 | 2021 Q1–2021 Q4 | 2022 Q1–2022 Q4 | 108 / 36 / 36 |

The gaps prevent future target information from entering the training data. We will use validation data for model selection and tuning and reserve testing data for the final evaluation.

## 3. Models and Baseline

We will evaluate the following models for all four prediction targets:

- Mean baseline
- Linear Regression
- Ridge Regression

The mean baseline predicts the average target value from the training data. It shows whether Linear and Ridge Regression provide better predictions. We will choose the Ridge regularization value using only the training and validation data. Time-series models are optional and will be compared with the required models.

## 4. Evaluation Metrics

We will report the same three metrics for every target and model:

- **Mean Absolute Error (MAE):** the average size of the prediction errors. Lower values are better.
- **Root Mean Squared Error (RMSE):** gives more weight to large prediction errors. Lower values are better.
- **R²:** compares the model with a simple average prediction. Higher values are better. A negative R² means that the model performed worse than the average prediction.


## 5. Preventing Data Leakage

We will follow these rules so that the models do not use information from the future:

- Use chronological training, validation, and testing periods.
- Use only information available at the prediction quarter.
- Learn missing-value rules, scaling, and feature selection from the training data only.
- Use validation data for model selection and Ridge tuning, while keeping testing data untouched until the final evaluation.

These steps will provide a more realistic measure of how the models may perform on future data.

## 6. Planned Comparisons and Stress Tests

We will use the following checks to understand how stable the model results are:

- Compare the mean baseline, Linear Regression, and Ridge Regression.
- Compare the two forecast horizons and prediction targets.
- Examine errors across counties and quarters.
- If possible, check whether pandemic-period observations affect the results.
- Compare the main feature set with a smaller set chosen using project knowledge.

These checks will help us identify where each model performs well and where its predictions are less reliable.

## 7. Results

We will add the MAE, RMSE, and R² results after the team finishes modeling. We will identify the best-performing models and summarize important error patterns.

## 8. Current Limitations

- The dataset includes only nine Ohio counties.
- The three-year targets have fewer usable observations than the one-year targets.
- Annual ACS and FHFA values are repeated across quarters.
- National and state variables repeat across counties.
- The pandemic period may affect housing and economic patterns.
- Some recent-quarter measures are based on partial-quarter data.
- Results may not apply to counties or time periods outside this dataset.

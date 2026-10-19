# Checkpoint 3: Chronological Data Splits

## Purpose

This project predicts future county-level ZHVI and ZORI growth. A random row
split would allow later housing outcomes to influence a model evaluated on
earlier quarters. `src/create_chronological_splits.py` instead creates fixed
chronological train, validation, and test datasets from
`data/processed/feature engineered data/master_county_quarter_features.csv`.

The script does not fit models, choose features, impute values, or modify raw
data. It writes three CSV files for each forecast horizon:

- `data/processed/1-year-ahead split/train.csv`
- `data/processed/1-year-ahead split/validation.csv`
- `data/processed/1-year-ahead split/test.csv`
- `data/processed/3-year-ahead split/train.csv`
- `data/processed/3-year-ahead split/validation.csv`
- `data/processed/3-year-ahead split/test.csv`

Run it from the repository root whenever the feature panel changes:

```bash
python src/create_chronological_splits.py
```

## Target-window rule

For a prediction quarter $t$ and forecast horizon $h$, the target is:

$$
g_{t,h} = 100\left(\frac{X_{t+h}}{X_t} - 1\right).
$$

A training row is eligible only when its target end quarter is strictly before
the validation start quarter:

$$
t + h < \text{validation start}.
$$

This prevents a training target from using an outcome that occurs during the
validation period. Rows between the latest eligible training quarter and the
validation start are deliberately omitted from training.

## Fixed schedules and sample sizes

| Horizon | Train prediction quarters | Validation prediction quarters | Test prediction quarters | Quarters: train / validation / test | County-quarter rows: train / validation / test |
| --- | --- | --- | --- | --- | --- |
| 1 year (4-quarter target) | 2015Q1–2019Q4 | 2021Q1–2022Q4 | 2023Q1–2024Q4 | 20 / 8 / 8 | 180 / 72 / 72 |
| 3 years (12-quarter target) | 2015Q1–2017Q4 | 2021Q1–2021Q4 | 2022Q1–2022Q4 | **12 / 4 / 4** | 108 / 36 / 36 |

Every split includes all nine project counties. Each horizon-specific CSV
contains only that horizon's two labels: one ZHVI target and one ZORI target.
The labels for the other horizon are removed so they cannot accidentally be
used as model inputs.

## Why validation and test files contain future targets

A validation row must retain its future target so that a completed model can be
evaluated after prediction. This is not leakage by itself. For example, the
one-year validation row for 2021Q1 uses 2022Q1 only as its held-out target.
The model is trained only on the separate training CSV, whose latest target
ends at 2020Q4.

For the three-year split, the final training prediction quarter is 2017Q4, and
its target ends in 2020Q4. Validation begins in 2021Q1. Thus no training target
window reaches the validation period.

## Safe modeling use

For one target, remove **both** remaining forward-looking target columns from
predictors. Fit every model-dependent operation on training data only:

```python
import pandas as pd

train_df = pd.read_csv("data/processed/1-year-ahead split/train.csv")
validation_df = pd.read_csv("data/processed/1-year-ahead split/validation.csv")
test_df = pd.read_csv("data/processed/1-year-ahead split/test.csv")

target_columns = [
    "zhvi_growth_ahead_1y_pct",
    "zori_growth_ahead_1y_pct",
]
feature_columns = [column for column in train_df.columns if column not in target_columns]

X_train = train_df[feature_columns]
y_train = train_df["zhvi_growth_ahead_1y_pct"]
X_validation = validation_df[feature_columns]
y_validation = validation_df["zhvi_growth_ahead_1y_pct"]
X_test = test_df[feature_columns]
y_test = test_df["zhvi_growth_ahead_1y_pct"]
```

Scalers, imputers, feature selection, and hyperparameter selection must use
training data only. Use validation data for model selection and keep test data
for the final evaluation.

## Limitations

These splits prevent target-window leakage; they do not prove that every
predictor was publicly available at the prediction quarter. ACS estimates have
publication delays, and the project must define a release-lag rule before
claiming a fully real-time model. The three-year model trains on 12 prediction
quarters (2015Q1–2017Q4), representing 108 county-quarter rows. Its estimates
and validation results will be materially less stable than the one-year model.

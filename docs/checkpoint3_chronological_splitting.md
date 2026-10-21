# Checkpoint 3: Leakage-Controlled One-Year-Ahead Data Split

## Purpose

This project models one-year-ahead county-level ZHVI and ZORI growth. The
canonical split files are created from
`data/processed/feature engineered data/master_county_quarter_features.csv` by
`src/create_chronological_splits.py`:

- `data/processed/1-year-ahead split/train.csv`
- `data/processed/1-year-ahead split/validation.csv`
- `data/processed/1-year-ahead split/test.csv`

Run the script from the repository root whenever the feature panel changes:

```bash
python src/create_chronological_splits.py
```

The script does not fit a model, impute data, or modify raw data.

## One-year target

For prediction quarter $t$, the one-year-ahead growth target is:

$$
g_{t,4} = 100\left(\frac{X_{t+4}}{X_t} - 1\right),
$$

where $X$ is either ZHVI or ZORI.

## Final schedule

| Dataset | Prediction origins | Target observations end by | Prediction quarters | County-quarter rows |
| --- | --- | --- | ---: | ---: |
| Train | 2015Q1–2019Q4 | 2020Q4 | 20 | 180 |
| Validation | 2021Q1–2021Q4 | 2022Q4 | 4 | 36 |
| Test | 2023Q1–2024Q4 | 2025Q4 | 8 | 72 |

Each file includes all nine project counties. The 180 / 36 / 72 row totals are
therefore 20 / 4 / 8 prediction quarters multiplied by nine counties.

The 2020 prediction-origin rows are intentionally not assigned to any split.
Several annual ACS variables are unavailable in 2020, so leaving the year as a
gap keeps those incomplete observations out of training and validation. The
2022 prediction-origin rows are also a gap: their one-year-ahead outcomes occur
during 2023, the test prediction year, and therefore cannot be used for model
selection.

## Leakage controls

The split applies two timing rules:

$$
\max(t_{\mathrm{train}}) + 4 < \min(t_{\mathrm{validation}})
$$

and

$$
\max(t_{\mathrm{validation}}) + 4 < \min(t_{\mathrm{test}}).
$$

Consequently, training targets end in 2020Q4 before validation starts in
2021Q1, and validation targets end in 2022Q4 before test prediction starts in
2023Q1. Model fitting and model selection therefore cannot use Zillow outcomes
from the test prediction period.

The split script also removes all three-year-ahead target columns from every
output file. They cannot become accidental one-year model predictors.

## Modeling use

Use the training data to fit preprocessing and candidate models. Use validation
data only to select among a small, pre-specified set of models. After selection,
refit the selected pipeline on the combined train and validation data, then
report final performance once on the untouched test data.

For either one-year target, remove **both** forward-looking one-year target
columns from predictors:

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
```

Fit scalers, imputers, feature selection, and model tuning on the training data
only. Do not inspect test metrics until the modeling choices are final.

## Why there is no three-year target split

The 2015–2025 panel does not contain enough earlier history to construct a
separate three-year train, validation, and test design in which training labels
precede validation prediction origins and validation labels precede test
prediction origins. A three-year split would require either earlier source data
or a weaker, non-real-time evaluation rule. The project therefore excludes the
three-year target from its current model-development workflow.

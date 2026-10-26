"""Create leakage-controlled one-year-ahead chronological data splits.

Run from the repository root:
    python src/create_chronological_splits.py

The script reads the feature-engineered county-quarter panel and writes the
one-year-ahead train, validation, and test CSV files. It does not change raw
data or fit a model.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# 1. PROJECT PATHS AND INPUT PANEL
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEATURE_PANEL_FILE = (
    PROCESSED
    / "feature engineered data"
    / "master_county_quarter_features.csv"
)
ONE_YEAR_OUTPUT = PROCESSED / "1-year-ahead split"

panel = pd.read_csv(FEATURE_PANEL_FILE, dtype={"county_fips": str})
panel["county_fips"] = panel["county_fips"].str.zfill(5)
panel = panel.sort_values(["year", "quarter", "county_fips"]).reset_index(drop=True)


# =============================================================================
# 2. VALIDATE THE COUNTY-QUARTER PANEL BEFORE SPLITTING
# =============================================================================

key_columns = ["county_fips", "year", "quarter"]
expected_counties = {
    "39017", "39035", "39049", "39061", "39095",
    "39113", "39151", "39153", "39165",
}

assert not panel.duplicated(key_columns).any(), "Duplicate county-quarter keys found."
assert set(panel["county_fips"]) == expected_counties, "Unexpected county FIPS codes found."
assert panel.groupby("county_fips").size().eq(44).all(), "Each county must have 44 quarters."

# This sortable number lets the timing checks compare quarters directly.
panel["_split_quarter_number"] = panel["year"] * 4 + (panel["quarter"] - 1)


# =============================================================================
# 3. DEFINE THE ONE-YEAR-AHEAD SPLIT SCHEDULE
# =============================================================================

# Training prediction origins: 2015Q1 through 2019Q4 (20 quarters, 180 rows).
# Their one-year-ahead targets end no later than 2020Q4.
# Calendar year 2020 is intentionally unused as the gap before validation,
# because several annual ACS variables are unavailable in that year.
validation_start = "2021Q1"
validation_end = "2021Q4"

# Validation prediction origins: 2021Q1 through 2021Q4 (4 quarters, 36 rows).
# Their one-year-ahead targets end no later than 2022Q4.
test_start = "2023Q1"
test_end = "2024Q4"

# The test has 8 prediction quarters (72 rows); its targets end in 2025Q4.
horizon_quarters = 4
target_columns = ["zhvi_growth_ahead_1y_pct", "zori_growth_ahead_1y_pct"]

validation_start_number = int(validation_start[:4]) * 4 + (int(validation_start[-1]) - 1)
validation_end_number = int(validation_end[:4]) * 4 + (int(validation_end[-1]) - 1)
test_start_number = int(test_start[:4]) * 4 + (int(test_start[-1]) - 1)
test_end_number = int(test_end[:4]) * 4 + (int(test_end[-1]) - 1)


# =============================================================================
# 4. BUILD THE THREE LEAKAGE-CONTROLLED DATASETS
# =============================================================================

# Remove every feature or label whose name describes a three-year historical
# or forward-looking housing-growth horizon. The project no longer evaluates
# three-year prediction, so these columns should not enter the one-year files
# as candidate predictors by accident.
#
# Do not remove every column containing the characters "3y":
# nat_treasury_3y_pct is an observed current-quarter Treasury yield, not a
# three-year-ahead housing feature, and remains eligible for one-year models.
three_year_housing_columns = [
    column
    for column in panel.columns
    if "prev_3y" in column or "ahead_3y" in column
]
columns_to_remove = three_year_housing_columns + ["_split_quarter_number"]
output_columns = [column for column in panel.columns if column not in columns_to_remove]

# Keep rows only when both one-year housing outcomes are observed. Predictor
# missingness is retained for later, model-specific handling.
observed_target_mask = panel[target_columns].notna().all(axis=1)

# Rule 1: every training target must end before validation starts.
train_mask = (
    (panel["_split_quarter_number"] + horizon_quarters < validation_start_number)
    & observed_target_mask
)

validation_mask = (
    panel["_split_quarter_number"].between(validation_start_number, validation_end_number)
    & observed_target_mask
)

test_mask = (
    panel["_split_quarter_number"].between(test_start_number, test_end_number)
    & observed_target_mask
)

train_data = panel.loc[train_mask, output_columns].copy()
validation_data = panel.loc[validation_mask, output_columns].copy()
test_data = panel.loc[test_mask, output_columns].copy()


# =============================================================================
# 5. VERIFY ROW SEPARATION, TARGET TIMING, AND COUNTY COVERAGE
# =============================================================================

train_keys = set(map(tuple, train_data[key_columns].to_numpy()))
validation_keys = set(map(tuple, validation_data[key_columns].to_numpy()))
test_keys = set(map(tuple, test_data[key_columns].to_numpy()))

assert not train_keys.intersection(validation_keys), "Train and validation rows overlap."
assert not train_keys.intersection(test_keys), "Train and test rows overlap."
assert not validation_keys.intersection(test_keys), "Validation and test rows overlap."

latest_train_origin = panel.loc[train_mask, "_split_quarter_number"].max()
latest_validation_origin = panel.loc[validation_mask, "_split_quarter_number"].max()

# Rule 1: fitting never uses targets from the validation period.
assert latest_train_origin + horizon_quarters < validation_start_number

# Rule 2: model selection never uses targets from the test prediction period.
assert latest_validation_origin + horizon_quarters < test_start_number

assert len(train_data) == 180, "Expected 180 train rows (20 quarters x 9 counties)."
assert len(validation_data) == 36, "Expected 36 validation rows (4 quarters x 9 counties)."
assert len(test_data) == 72, "Expected 72 test rows (8 quarters x 9 counties)."
assert set(train_data["county_fips"]) == expected_counties
assert set(validation_data["county_fips"]) == expected_counties
assert set(test_data["county_fips"]) == expected_counties

# The one-year modeling files must contain no three-year historical features or
# forward-looking three-year labels. This protects downstream notebooks that
# choose numeric predictors programmatically.
for split_name, split_data in {
    "train": train_data,
    "validation": validation_data,
    "test": test_data,
}.items():
    disallowed_columns = [
        column
        for column in split_data.columns
        if "prev_3y" in column or "ahead_3y" in column
    ]
    assert not disallowed_columns, (
        f"{split_name} includes three-year modeling columns: {disallowed_columns}"
    )


# =============================================================================
# 6. SAVE THE CANONICAL ONE-YEAR-AHEAD FILES
# =============================================================================

ONE_YEAR_OUTPUT.mkdir(parents=True, exist_ok=True)
train_data.to_csv(ONE_YEAR_OUTPUT / "train.csv", index=False)
validation_data.to_csv(ONE_YEAR_OUTPUT / "validation.csv", index=False)
test_data.to_csv(ONE_YEAR_OUTPUT / "test.csv", index=False)

print("1-year-ahead split")
print(f"  train: {len(train_data)} rows (2015Q1-2019Q4)")
print(f"  validation: {len(validation_data)} rows (2021Q1-2021Q4)")
print(f"  test: {len(test_data)} rows (2023Q1-2024Q4)")

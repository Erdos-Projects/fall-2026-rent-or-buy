"""Create fixed chronological modeling splits for one- and three-year targets.

Run from the repository root:
    python src/create_chronological_splits.py

The script reads the feature-engineered county-quarter panel and writes three
CSV files (train, validation, and test) for each forecast horizon. It does not
change raw data or fit a model.
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
THREE_YEAR_OUTPUT = PROCESSED / "3-year-ahead split"

panel = pd.read_csv(FEATURE_PANEL_FILE, dtype={"county_fips": str})
panel["county_fips"] = panel["county_fips"].str.zfill(5)
panel = panel.sort_values(["year", "quarter", "county_fips"]).reset_index(drop=True)


# =============================================================================
# 2. VALIDATE THE COUNTY-QUARTER PANEL BEFORE SPLITTING
# =============================================================================

key_columns = ["county_fips", "year", "quarter"]
expected_counties = {"39017", "39035", "39049", "39061", "39095", "39113", "39151", "39153", "39165"}

assert not panel.duplicated(key_columns).any(), "Duplicate county-quarter keys found."
assert set(panel["county_fips"]) == expected_counties, "Unexpected county FIPS codes found."
assert panel.groupby("county_fips").size().eq(44).all(), "Each county must have 44 quarters."

# A sortable quarter number supports the strict target-window rule below.
panel["_split_quarter_number"] = panel["year"] * 4 + (panel["quarter"] - 1)


# =============================================================================
# 3. DEFINE THE FIXED CHRONOLOGICAL SCHEDULES
# =============================================================================

split_schedules = [
    {
        "label": "1-year-ahead",
        "horizon_quarters": 4,
        "target_columns": ["zhvi_growth_ahead_1y_pct", "zori_growth_ahead_1y_pct"],
        "validation_start": "2021Q1",
        "validation_end": "2022Q4",
        "test_start": "2023Q1",
        "test_end": "2024Q4",
        "output_folder": ONE_YEAR_OUTPUT,
    },
    {
        "label": "3-year-ahead",
        "horizon_quarters": 12,
        "target_columns": ["zhvi_growth_ahead_3y_pct", "zori_growth_ahead_3y_pct"],
        "validation_start": "2021Q1",
        "validation_end": "2021Q4",
        "test_start": "2022Q1",
        "test_end": "2022Q4",
        "output_folder": THREE_YEAR_OUTPUT,
    },
]


# =============================================================================
# 4. CREATE AND SAVE TRAIN, VALIDATION, AND TEST DATASETS
# =============================================================================

all_target_columns = [
    "zhvi_growth_ahead_1y_pct",
    "zori_growth_ahead_1y_pct",
    "zhvi_growth_ahead_3y_pct",
    "zori_growth_ahead_3y_pct",
]

for schedule in split_schedules:
    validation_start_year = int(schedule["validation_start"][:4])
    validation_start_quarter = int(schedule["validation_start"][-1])
    validation_end_year = int(schedule["validation_end"][:4])
    validation_end_quarter = int(schedule["validation_end"][-1])
    test_start_year = int(schedule["test_start"][:4])
    test_start_quarter = int(schedule["test_start"][-1])
    test_end_year = int(schedule["test_end"][:4])
    test_end_quarter = int(schedule["test_end"][-1])

    validation_start_number = validation_start_year * 4 + (validation_start_quarter - 1)
    validation_end_number = validation_end_year * 4 + (validation_end_quarter - 1)
    test_start_number = test_start_year * 4 + (test_start_quarter - 1)
    test_end_number = test_end_year * 4 + (test_end_quarter - 1)

    # Keep only the two targets for this horizon. The other horizon's labels
    # are removed so they cannot accidentally become model input features.
    columns_to_remove = [
        column for column in all_target_columns if column not in schedule["target_columns"]
    ] + ["_split_quarter_number"]
    output_columns = [column for column in panel.columns if column not in columns_to_remove]

    # A row is usable only when both same-horizon housing targets are observed.
    # Predictor missingness remains unchanged for later model-specific handling.
    observed_target_mask = panel[schedule["target_columns"]].notna().all(axis=1)

    # Strict leakage rule: target_end_quarter must be before validation starts.
    train_mask = (
        (panel["_split_quarter_number"] + schedule["horizon_quarters"] < validation_start_number)
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

    # Verify that no county-quarter row appears in more than one output file.
    train_keys = set(map(tuple, train_data[key_columns].to_numpy()))
    validation_keys = set(map(tuple, validation_data[key_columns].to_numpy()))
    test_keys = set(map(tuple, test_data[key_columns].to_numpy()))
    assert not train_keys.intersection(validation_keys), "Train and validation rows overlap."
    assert not train_keys.intersection(test_keys), "Train and test rows overlap."
    assert not validation_keys.intersection(test_keys), "Validation and test rows overlap."

    # Verify strict target-window timing and county coverage.
    latest_train_number = panel.loc[train_mask, "_split_quarter_number"].max()
    assert latest_train_number + schedule["horizon_quarters"] < validation_start_number
    assert set(train_data["county_fips"]) == expected_counties
    assert set(validation_data["county_fips"]) == expected_counties
    assert set(test_data["county_fips"]) == expected_counties

    schedule["output_folder"].mkdir(parents=True, exist_ok=True)
    train_data.to_csv(schedule["output_folder"] / "train.csv", index=False)
    validation_data.to_csv(schedule["output_folder"] / "validation.csv", index=False)
    test_data.to_csv(schedule["output_folder"] / "test.csv", index=False)

    print(f"{schedule['label']} split")
    print(f"  train: {len(train_data)} rows")
    print(f"  validation: {len(validation_data)} rows")
    print(f"  test: {len(test_data)} rows")

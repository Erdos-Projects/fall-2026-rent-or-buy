"""Extend the financial calculator to 2025 origins using the selected models.

Run from any directory: python src/build_calculator_forecasts.py
No model selection, test-set fitting, or future-outcome imputation occurs here.
"""

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 1. INPUTS AND THE EXISTING MODEL-COMPARISON SPECIFICATION
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/processed/1-year-ahead split"
PANEL = ROOT / "data/processed/feature engineered data/master_county_quarter_features.csv"
COMPARISON = ROOT / "results/checkpoint3_model_compare"
OUTPUT = ROOT / "results/financial_calculator"
HTML = ROOT / "financial_calculator.html"
KEYS = ["county_fips", "year", "quarter"]

# Keep the same ten numeric predictors and county fixed effects used in the
# published growth-model comparison. ACS is not in this selected feature set.
PREDICTORS = [
    "zhvi_growth_qoq_pct", "zhvi_growth_prev_1y_pct",
    "zori_growth_qoq_pct", "zori_growth_prev_1y_pct",
    "nat_mortgage_rate_30y_pct", "mortgage_rate_change_prev_1y_pp",
    "nat_inflation_yoy_pct", "cty_unemployment_rate_pct",
    "cty_unemployment_change_prev_1y_pp", "price_to_rent_ratio",
]
SETTINGS = {
    "ZHVI": {"model": "Ridge", "alpha": 100.0, "quarter_terms": True,
             "level": "zhvi_usd", "growth": "zhvi_growth_ahead_1y_pct"},
    "ZORI": {"model": "Elastic Net", "alpha": 0.1, "l1_ratio": 0.75,
             "quarter_terms": False, "level": "zori_usd_month",
             "growth": "zori_growth_ahead_1y_pct"},
}
PARTIAL_FLAGS = ["cty_partial_quarter", "nat_inflation_yoy_pct_partial"]
inputs = {
    "train": SPLITS / "train.csv",
    "validation": SPLITS / "validation.csv",
    "test": SPLITS / "test.csv",
    "panel": PANEL,
    "selection": COMPARISON / "validation_model_comparison.csv",
    "saved_predictions": COMPARISON / "one_year_ahead_growth_test_predictions.csv",
}
frames = {name: pd.read_csv(path, dtype={"county_fips": str})
          for name, path in inputs.items()}


# =============================================================================
# 2. FREEZE THE FITTING SAMPLE AND CHECK THE NEW FORECAST INPUTS
# =============================================================================
# This is exactly the final fitting sample used for the published test results:
# 2015-2019 training origins plus 2021 validation origins (216 county-quarters).
# Do not refit on the 2023-2024 test outcomes just to extend the calculator.
training = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
assert len(training) == 216
assert set(training["year"]) == {2015, 2016, 2017, 2018, 2019, 2021}
assert not training.duplicated(KEYS).any()
forecast = frames["panel"].loc[frames["panel"]["year"].eq(2025)].copy()
forecast = forecast.sort_values(KEYS).reset_index(drop=True)
assert len(forecast) == 36 and not forecast.duplicated(KEYS).any()
assert forecast.groupby("quarter").size().eq(9).all()
assert set(forecast["county_fips"]) == set(training["county_fips"])
assert forecast[PREDICTORS].notna().all().all(), "A 2025 predictor is missing."
assert np.isfinite(forecast[PREDICTORS].to_numpy()).all()
assert forecast[[s["growth"] for s in SETTINGS.values()]].isna().all().all()
assert forecast[[s["level"] for s in SETTINGS.values()]].gt(0).all().all()
assert forecast[["nat_mortgage_rate_30y_pct", "nat_treasury_1y_pct"]].notna().all().all()


# =============================================================================
# 3. REBUILD EACH SELECTED PIPELINE AND REPRODUCE ITS SAVED TEST PREDICTIONS
# =============================================================================
outputs = []
verification = {}
for target, setting in SETTINGS.items():
    selected = frames["selection"].loc[
        frames["selection"]["target"].eq(target)
        & frames["selection"]["model"].eq(setting["model"])
        & frames["selection"]["fit_target"].eq("1-year-ahead growth")
    ]
    assert len(selected) == 1
    selected = selected.iloc[0]
    assert (selected["quarter_terms"] == "Yes") == setting["quarter_terms"]
    assert float(re.search(r"alpha=([\d.]+)", selected["specification"])[1]) == setting["alpha"]
    if target == "ZORI":
        assert float(re.search(r"l1_ratio=([\d.]+)", selected["specification"])[1]) == setting["l1_ratio"]

    # Medians and standardization are learned from the fitting sample only.
    # Early training rows have missing lags; the 2025 predictors need no fill.
    transformations = [
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), PREDICTORS),
        ("county_fixed_effects", OneHotEncoder(drop="first", handle_unknown="ignore"),
         ["county_name"]),
    ]
    if setting["quarter_terms"]:
        transformations.append((
            "quarter_fixed_effects",
            OneHotEncoder(categories=[[1, 2, 3, 4]], drop="first", handle_unknown="ignore"),
            ["quarter"],
        ))
    estimator = (Ridge(alpha=setting["alpha"]) if target == "ZHVI" else
                 ElasticNet(alpha=setting["alpha"], l1_ratio=setting["l1_ratio"], max_iter=20000))
    model = Pipeline([
        ("preprocessing", ColumnTransformer(transformations, remainder="drop")),
        ("model", estimator),
    ])
    model.fit(training, training[setting["growth"]])

    # This independent reproduction guards against accidentally changing the
    # feature set, fitting rows, regularization, or categorical coding.
    saved = frames["saved_predictions"].loc[
        frames["saved_predictions"]["target"].eq(target)
        & frames["saved_predictions"]["model"].eq(setting["model"])
        & frames["saved_predictions"]["fit_target"].eq("1-year-ahead growth")
    ]
    check = frames["test"].merge(saved[KEYS + ["predicted_growth_pct"]],
                                 on=KEYS, validate="one_to_one")
    assert len(check) == 72
    difference = np.abs(model.predict(check) - check["predicted_growth_pct"])
    assert difference.max() < 1e-8, "Rebuilt model differs from published predictions."
    verification[target] = {"test_rows": len(check),
                            "max_absolute_growth_difference_pp": float(difference.max())}

    prediction = model.predict(forecast)
    assert np.isfinite(prediction).all() and (prediction > -100).all()
    result = forecast[KEYS + ["county_name"] + PARTIAL_FLAGS].copy()
    result["target"] = target
    result["model"] = setting["model"]
    result["quarter_terms"] = setting["quarter_terms"]
    result["forecast_origin"] = result["year"].astype(str) + "Q" + result["quarter"].astype(str)
    result["target_period"] = (result["year"] + 1).astype(str) + "Q" + result["quarter"].astype(str)
    result["current_level"] = forecast[setting["level"]]
    result["predicted_growth_pct"] = prediction
    result["predicted_level_t_plus_4"] = result["current_level"] * (1 + prediction / 100)
    result["actual_growth_pct"] = np.nan
    result["actual_level_t_plus_4"] = np.nan
    result["outcome_status"] = "not_observed_in_project_data"
    outputs.append(result)


# =============================================================================
# 4. SAVE FORECASTS AND AUDIT METADATA; UPDATE ONLY THE HTML'S 2025 FORECASTS
# =============================================================================
predictions = pd.concat(outputs, ignore_index=True)
assert len(predictions) == 72
assert not predictions.duplicated(KEYS + ["target"]).any()
html = HTML.read_text()
match = re.search(r"const DATA = (\[.*?\]);", html)
assert match, "The embedded calculator DATA array was not found."
embedded = json.loads(match[1])
lookup = predictions.set_index(["county_name", "year", "quarter", "target"])
updated = 0
for record in embedded:
    county, year, quarter = record[:3]
    if year != 2025:
        continue
    assert record[7] is None and record[8] is None, "Do not replace observed outcomes."
    source = forecast.loc[forecast["county_name"].eq(county) & forecast["quarter"].eq(quarter)].iloc[0]
    for position, column in [(5, "nat_mortgage_rate_30y_pct"), (6, "nat_treasury_1y_pct")]:
        assert abs(record[position] - source[column]) < 0.00051, "Embedded rate differs from the source panel."
    for target, position in [("ZHVI", 9), ("ZORI", 10)]:
        value = lookup.loc[(county, year, quarter, target)]
        # Embedded starting levels are rounded to cents; verify alignment.
        assert abs(record[3 if target == "ZHVI" else 4] - value["current_level"]) < 0.011
        record[position] = round(float(value["predicted_growth_pct"]), 6)
    updated += 1
assert updated == 36

OUTPUT.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUTPUT / "forecasts_2025.csv", index=False)
manifest = {
    "forecast_origins": "2025Q1-2025Q4", "target_periods": "2026Q1-2026Q4",
    "fitting_origins": "2015Q1-2019Q4 and 2021Q1-2021Q4",
    "fitting_rows": len(training), "forecast_county_quarters": len(forecast),
    "models": SETTINGS, "numeric_predictors": PREDICTORS,
    "test_reproduction": verification,
    "2025_predictor_missing_cells": int(forecast[PREDICTORS].isna().sum().sum()),
    "partial_quarter": "2025Q4 county labor and inflation use November-December inputs.",
    "data_vintage_note": "Retrospective project snapshots; not a real-time, as-of-release backtest.",
    "outcomes": "2026 outcomes unavailable in the project; not imputed or scored.",
    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in inputs.values()},
}
(OUTPUT / "forecast_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
payload = json.dumps(embedded, separators=(",", ":"), allow_nan=False)
HTML.write_text(html[:match.start(1)] + payload + html[match.end(1):])
print(f"Saved {len(predictions)} forecasts; updated {updated} calculator county-quarters.")
print(json.dumps(verification, indent=2))

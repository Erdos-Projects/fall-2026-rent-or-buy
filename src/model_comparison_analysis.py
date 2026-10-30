"""Run the Checkpoint 3 pooled and time-series model comparison.

Run this file from the repository's ``notebooks`` directory (or run the
companion notebook). Outputs are written to results/checkpoint3_model_compare.
Model specifications are selected on the 2021 validation period. The explicit
growth-only eligibility rule was documented after earlier test results were
seen, so the existing test scores are descriptive for that revised rule.
"""

from pathlib import Path
import ast
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA


# -----------------------------------------------------------------------------
# 1. Paths, targets, and fixed comparison design
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "1-year-ahead split"
FEATURE_PANEL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "feature engineered data"
    / "master_county_quarter_features.csv"
)
DICTIONARY_PATH = PROJECT_ROOT / "data" / "metadata" / "master_feature_dictionary.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "checkpoint3_model_compare"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PR #15 already evaluated the unchanged ARIMA/ARIMAX specifications. Reuse
# those checked-in results by default; --recompute-time-series runs the full
# (much slower) rolling-origin search and forecasting on a fresh dataset.
RECOMPUTE_TIME_SERIES = "--recompute-time-series" in sys.argv
CACHE_TIME_SERIES = (
    not RECOMPUTE_TIME_SERIES
    and all((RESULTS_DIR / name).exists() for name in [
        "arima_order_and_exogenous_selection.csv", "predictions.csv",
        "test_confidence_bands.csv", "one_year_ahead_growth_test_bands.csv",
    ])
)
if CACHE_TIME_SERIES:
    cached_ts_candidates = pd.read_csv(RESULTS_DIR / "arima_order_and_exogenous_selection.csv")
    cached_ts_predictions = pd.read_csv(RESULTS_DIR / "predictions.csv", dtype={"county_fips": str})
    cached_ts_level_bands = pd.read_csv(RESULTS_DIR / "test_confidence_bands.csv")
    cached_ts_growth_bands = pd.read_csv(RESULTS_DIR / "one_year_ahead_growth_test_bands.csv")

LOCAL_PYTHON_PACKAGES = PROJECT_ROOT.parent / ".python_packages"
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))
from xgboost import XGBRegressor

TARGETS = {
    "ZHVI": {"growth": "zhvi_growth_ahead_1y_pct", "level": "zhvi_usd"},
    "ZORI": {"growth": "zori_growth_ahead_1y_pct", "level": "zori_usd_month"},
}
COUNTY_KEY = "county_fips"
COUNTY_COLUMN = "county_name"
HORIZON_QUARTERS = 4
GROWTH_FIT = "1-year-ahead growth"
LEVEL_FIT = "Original level at t+4"
FIT_TARGETS = [GROWTH_FIT, LEVEL_FIT]

# One fixed, domain-informed feature set is shared by every pooled model that
# uses predictors. Categorical county/quarter effects are added separately.
COMMON_PREDICTORS = [
    "zhvi_growth_qoq_pct", "zhvi_growth_prev_1y_pct",
    "zori_growth_qoq_pct", "zori_growth_prev_1y_pct",
    "nat_mortgage_rate_30y_pct", "mortgage_rate_change_prev_1y_pp",
    "nat_inflation_yoy_pct", "cty_unemployment_rate_pct",
    "cty_unemployment_change_prev_1y_pp", "price_to_rent_ratio",
]

# Required comparison order. Baseline display names depend on fitting target.
PREDICTOR_MODELS = [
    "Ordinary Linear Regression", "LASSO", "Elastic Net", "Ridge", "XGBoost",
]
MODEL_ORDER = [
    "Mean growth baseline", "Mean level baseline",
    "Ordinary Linear Regression", "LASSO", "Elastic Net", "Ridge", "XGBoost",
    "ARIMA", "ARIMAX",
]

# The project's forecasting question is about one-year growth. Direct-level
# fits remain diagnostic comparisons, but only growth-fitted pooled models
# are eligible for the tentative model choice. Because this policy was
# formalized after inspecting the prior comparison, treat test scores as
# descriptive rather than a pristine evaluation of the revised selection rule.
ELIGIBLE_FIT_TARGET = GROWTH_FIT

REGRESSION_CANDIDATES = {
    "Ordinary Linear Regression": [LinearRegression()],
    "LASSO": [Lasso(alpha=a, max_iter=20000) for a in [0.01, 0.1, 1.0, 10.0]],
    "Elastic Net": [
        ElasticNet(alpha=a, l1_ratio=r, max_iter=20000)
        for a in [0.01, 0.1, 1.0] for r in [0.25, 0.5, 0.75]
    ],
    "Ridge": [Ridge(alpha=a) for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]],
    "XGBoost": [
        XGBRegressor(
            objective="reg:squarederror", random_state=42, n_jobs=1,
            n_estimators=trees, max_depth=depth, learning_rate=rate,
            min_child_weight=child, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=regularization,
        )
        for trees, depth, rate, child, regularization in [
            (50, 1, 0.05, 10, 10), (100, 1, 0.05, 10, 10),
            (75, 2, 0.03, 15, 20),
        ]
    ],
}


# -----------------------------------------------------------------------------
# 2. Read the already-defined chronological splits and check panel structure
# -----------------------------------------------------------------------------
train = pd.read_csv(SPLIT_DIR / "train.csv", dtype={COUNTY_KEY: str})
validation = pd.read_csv(SPLIT_DIR / "validation.csv", dtype={COUNTY_KEY: str})
test = pd.read_csv(SPLIT_DIR / "test.csv", dtype={COUNTY_KEY: str})
feature_panel = pd.read_csv(FEATURE_PANEL_PATH, dtype={COUNTY_KEY: str})
feature_dictionary = pd.read_csv(DICTIONARY_PATH)

for frame in [train, validation, test, feature_panel]:
    frame[COUNTY_KEY] = frame[COUNTY_KEY].astype(str).str.zfill(5)
    for target_name, info in TARGETS.items():
        # The 1-year target is an observed growth label. This reconstructs its
        # realized t+4 level for direct-level model fitting and common scoring.
        frame[f"{target_name.lower()}_level_t_plus_4"] = (
            frame[info["level"]] * (1 + frame[info["growth"]] / 100)
        )

for frame, expected_rows, label in [
    (train, 180, "train"), (validation, 36, "validation"), (test, 72, "test")
]:
    assert len(frame) == expected_rows, f"Unexpected {label} size: {len(frame)}"
    assert frame.groupby(["year", "quarter"]).size().eq(9).all()
assert feature_panel.groupby(COUNTY_KEY).size().eq(44).all()
assert len(feature_panel) == 396
assert set(COMMON_PREDICTORS).issubset(train.columns)

feature_panel["period"] = pd.PeriodIndex(
    feature_panel["year"].astype(str) + "Q" + feature_panel["quarter"].astype(str),
    freq="Q",
)


# -----------------------------------------------------------------------------
# 3. Shared pooled model helpers: same counties, features, and scoring scale
# -----------------------------------------------------------------------------
def score(actual, predicted):
    """Return standard out-of-sample metrics for a common observed scale."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def level_target_column(target_name):
    return f"{target_name.lower()}_level_t_plus_4"


def display_name(model_name, fit_target):
    if model_name == "Mean baseline":
        return "Mean growth baseline" if fit_target == GROWTH_FIT else "Mean level baseline"
    return model_name


def make_pooled_pipeline(numeric_features, estimator, include_quarter):
    """One pooled fit, with county fixed effects and optional Q2-Q4 effects."""
    pieces = [
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("county_fixed_effects", OneHotEncoder(drop="first", handle_unknown="ignore"), [COUNTY_COLUMN]),
    ]
    if include_quarter:
        pieces.append((
            "quarter_fixed_effects",
            OneHotEncoder(categories=[[1, 2, 3, 4]], drop="first", handle_unknown="ignore"),
            ["quarter"],
        ))
    return Pipeline([
        ("preprocessing", ColumnTransformer(pieces, remainder="drop")),
        ("model", estimator),
    ])


def fit_predict_pooled(training, prediction, target_name, fit_target,
                       model_name, estimator, include_quarter):
    """Fit a model to growth or level and return predictions on level scale."""
    is_growth_fit = fit_target == GROWTH_FIT
    outcome = TARGETS[target_name]["growth"] if is_growth_fit else level_target_column(target_name)
    features = [] if model_name == "Mean baseline" else COMMON_PREDICTORS.copy()
    if features and not is_growth_fit:
        # A direct level fit must see the current level. Growth-fit models use
        # that same observed level when translating growth to future dollars.
        features.append(TARGETS[target_name]["level"])
    columns = features + [COUNTY_COLUMN] + (["quarter"] if include_quarter else [])
    model = make_pooled_pipeline(features, estimator, include_quarter)
    model.fit(training[columns], training[outcome])
    raw_prediction = model.predict(prediction[columns])
    if is_growth_fit:
        level_prediction = prediction[TARGETS[target_name]["level"]].to_numpy() * (1 + raw_prediction / 100)
    else:
        level_prediction = raw_prediction
    return model, np.asarray(level_prediction), np.asarray(raw_prediction)


def fit_description(estimator, fit_target):
    """Compact parameter note that also makes each model's fitted target explicit."""
    if isinstance(estimator, ElasticNet):
        setting = f"alpha={estimator.alpha:g}, l1_ratio={estimator.l1_ratio:g}"
    elif hasattr(estimator, "alpha"):
        setting = f"alpha={estimator.alpha:g}"
    elif isinstance(estimator, XGBRegressor):
        setting = (
            f"trees={estimator.n_estimators}, depth={estimator.max_depth}, "
            f"learning_rate={estimator.learning_rate:g}"
        )
    else:
        setting = "unregularized OLS"
    return f"fit={fit_target}; {setting}"


# Evaluate every pooled family with/without quarter terms. Candidate parameter
# choices and quarter terms are selected on the 2021 validation period only.
validation_rows = []
validation_predictions = {}
selected_pooled = {}
for target_name in TARGETS:
    for fit_target in FIT_TARGETS:
        actual_level = validation[level_target_column(target_name)].to_numpy()
        for include_quarter in [False, True]:
            # Distinct growth and level baselines each have no numeric predictors.
            model_name = "Mean baseline"
            _, prediction, _ = fit_predict_pooled(
                train, validation, target_name, fit_target, model_name,
                LinearRegression(), include_quarter,
            )
            label = display_name(model_name, fit_target)
            validation_predictions[(target_name, label)] = prediction
            validation_rows.append({
                "target": target_name, "model": label, "fit_target": fit_target,
                "quarter_terms": "Yes" if include_quarter else "No",
                "specification": "county mean growth" if fit_target == GROWTH_FIT else "county mean level",
                **score(actual_level, prediction),
            })

            for model_name, estimators in REGRESSION_CANDIDATES.items():
                candidates = []
                for estimator in estimators:
                    _, candidate_prediction, _ = fit_predict_pooled(
                        train, validation, target_name, fit_target, model_name,
                        clone(estimator), include_quarter,
                    )
                    metrics = score(actual_level, candidate_prediction)
                    candidates.append((metrics["RMSE"], metrics["MAE"], estimator,
                                       candidate_prediction, metrics))
                candidates.sort(key=lambda row: (row[0], row[1]))
                _, _, chosen_estimator, chosen_prediction, chosen_metrics = candidates[0]
                label = model_name
                validation_predictions[(target_name, f"{label}|{fit_target}|{include_quarter}")] = chosen_prediction
                validation_rows.append({
                    "target": target_name, "model": label, "fit_target": fit_target,
                    "quarter_terms": "Yes" if include_quarter else "No",
                    "specification": fit_description(chosen_estimator, fit_target),
                    **chosen_metrics,
                })
                selected_pooled[(target_name, label, fit_target, include_quarter)] = clone(chosen_estimator)


# -----------------------------------------------------------------------------
# 4. County-specific ARIMA and ARIMAX candidates, selected on validation only
# -----------------------------------------------------------------------------
ARIMA_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
ARIMAX_BUNDLES = {
    "Mortgage rate": ["nat_mortgage_rate_30y_pct"],
    "Mortgage + inflation": ["nat_mortgage_rate_30y_pct", "nat_inflation_yoy_pct"],
    "Mortgage + county unemployment": ["nat_mortgage_rate_30y_pct", "cty_unemployment_rate_pct"],
}
ARIMAX_COLUMNS = sorted({x for values in ARIMAX_BUNDLES.values() for x in values})
county_histories = {}
for county_id, history in feature_panel.groupby(COUNTY_KEY):
    history = history.sort_values("period").set_index("period")
    county_histories[county_id] = history[["zhvi_usd", "zori_usd_month"] + ARIMAX_COLUMNS].copy()


def make_arima_exog(history, origin, exogenous_columns, include_quarter):
    observed = history.loc[:origin]
    future_periods = pd.period_range(origin + 1, periods=HORIZON_QUARTERS, freq="Q")
    observed_parts, future_parts = [], []
    if exogenous_columns:
        x = observed[exogenous_columns].astype(float)
        if x.isna().any().any():
            raise ValueError("Missing ARIMAX training predictor")
        mean = x.mean()
        scale = x.std(ddof=0).replace(0, np.nan)
        if scale.isna().any():
            raise ValueError("Constant ARIMAX predictor")
        standardized = (x - mean) / scale
        future = pd.DataFrame(
            np.repeat(standardized.iloc[[-1]].to_numpy(), HORIZON_QUARTERS, axis=0),
            index=future_periods, columns=exogenous_columns,
        )
        observed_parts.append(standardized)
        future_parts.append(future)
    if include_quarter:
        qo = pd.get_dummies(observed.index.quarter, prefix="Q", dtype=float)
        qo.index = observed.index
        qf = pd.get_dummies(future_periods.quarter, prefix="Q", dtype=float)
        qf.index = future_periods
        observed_parts.append(qo.reindex(columns=["Q_2", "Q_3", "Q_4"], fill_value=0.0))
        future_parts.append(qf.reindex(columns=["Q_2", "Q_3", "Q_4"], fill_value=0.0))
    if not observed_parts:
        return None, None
    return pd.concat(observed_parts, axis=1), pd.concat(future_parts, axis=1)


def forecast_one(history, origin, level_column, order, exogenous_columns, include_quarter):
    y = history.loc[:origin, level_column].astype(float) / 1000.0
    observed_exog, future_exog = make_arima_exog(history, origin, exogenous_columns, include_quarter)
    trend = "t" if order[1] == 1 else "c"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(
            y, exog=observed_exog, order=order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit()
        forecast = fitted.get_forecast(steps=HORIZON_QUARTERS, exog=future_exog)
    mean = np.asarray(forecast.predicted_mean, dtype=float)
    interval = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
    return {"prediction": mean[-1] * 1000, "lower_95": interval[-1, 0] * 1000,
            "upper_95": interval[-1, 1] * 1000}


def forecast_rows(rows, target_name, order, exogenous_columns, include_quarter):
    column = TARGETS[target_name]["level"]
    output = []
    for row in rows.itertuples(index=False):
        origin = pd.Period(year=int(row.year), quarter=int(row.quarter), freq="Q")
        forecast = forecast_one(
            county_histories[row.county_fips], origin, column, order,
            exogenous_columns, include_quarter,
        )
        output.append({COUNTY_KEY: row.county_fips, "year": int(row.year),
                       "quarter": int(row.quarter), **forecast})
    return pd.DataFrame(output)


ts_candidates = []
selected_ts = {}
if CACHE_TIME_SERIES:
    # The cached candidate table is from PR #15. Its input splits and
    # time-series model definitions are unchanged in this PR.
    ts_candidate_df = cached_ts_candidates.rename(columns={
        "family": "model", "quarter_dummies": "quarter_terms",
    }).copy()
    if ts_candidate_df["quarter_terms"].dtype == bool:
        ts_candidate_df["quarter_terms"] = ts_candidate_df["quarter_terms"].map(
            {True: "Yes", False: "No"}
        )
    assert set(ts_candidate_df["quarter_terms"].dropna()).issubset({"Yes", "No"})
    ts_candidate_df["fit_target"] = LEVEL_FIT
    for (target_name, family, quarter_terms), group in ts_candidate_df.groupby(
        ["target", "model", "quarter_terms"]
    ):
        selected = group.sort_values(["RMSE", "MAE"]).iloc[0]
        bundle = selected["predictor_bundle"]
        selected_ts[(target_name, family, quarter_terms == "Yes")] = {
            "order": ast.literal_eval(selected["order"]),
            "bundle_name": "None" if pd.isna(bundle) else bundle,
            "exogenous": [] if pd.isna(selected["predictors"])
            else [value.strip() for value in selected["predictors"].split(",")],
        }
else:
    for target_name in TARGETS:
        actual = validation[level_target_column(target_name)].to_numpy()
        for family in ["ARIMA", "ARIMAX"]:
            bundles = {"None": []} if family == "ARIMA" else ARIMAX_BUNDLES
            for include_quarter in [False, True]:
                possible = []
                for bundle_name, exogenous in bundles.items():
                    for order in ARIMA_ORDERS:
                        try:
                            forecasts = forecast_rows(validation, target_name, order, exogenous, include_quarter)
                            metrics = score(actual, forecasts["prediction"].to_numpy())
                            row = {
                                "target": target_name, "model": family, "fit_target": LEVEL_FIT,
                                "quarter_terms": "Yes" if include_quarter else "No",
                                "order": str(order), "p": order[0], "d": order[1], "q": order[2],
                                "predictor_bundle": bundle_name, "predictors": ", ".join(exogenous),
                                **metrics,
                            }
                            possible.append((metrics["RMSE"], metrics["MAE"], order, bundle_name, exogenous, row))
                        except Exception as err:
                            ts_candidates.append({
                                "target": target_name, "model": family, "fit_target": LEVEL_FIT,
                                "quarter_terms": "Yes" if include_quarter else "No",
                                "order": str(order), "p": order[0], "d": order[1], "q": order[2],
                                "predictor_bundle": bundle_name, "predictors": ", ".join(exogenous),
                                "MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                                "fit_error": f"{type(err).__name__}: {err}",
                            })
                possible.sort(key=lambda item: (item[0], item[1]))
                best = possible[0]
                ts_candidates.extend([item[-1] for item in possible])
                selected_ts[(target_name, family, include_quarter)] = {
                    "order": best[2], "bundle_name": best[3], "exogenous": best[4],
                }
    ts_candidate_df = pd.DataFrame(ts_candidates)


# -----------------------------------------------------------------------------
# 5. Choose each model's seasonal version and the target-specific winners
# -----------------------------------------------------------------------------
ts_validation_rows = []
for target_name in TARGETS:
    for family in ["ARIMA", "ARIMAX"]:
        for include_quarter in [False, True]:
            candidates = ts_candidate_df[
                ts_candidate_df["target"].eq(target_name)
                & ts_candidate_df["model"].eq(family)
                & ts_candidate_df["quarter_terms"].eq("Yes" if include_quarter else "No")
                & ts_candidate_df["RMSE"].notna()
            ].sort_values(["RMSE", "MAE"])
            selected = candidates.iloc[0]
            ts_validation_rows.append({
                "target": target_name, "model": family, "fit_target": LEVEL_FIT,
                "quarter_terms": selected["quarter_terms"],
                "specification": f"ARIMA{selected['order']}; {selected['predictor_bundle'] if pd.notna(selected['predictor_bundle']) else 'None'}",
                "RMSE": selected["RMSE"], "MAE": selected["MAE"], "R2": selected["R2"],
            })

# Each pooled row already contains the best hyperparameter for that seasonal
# setting. Keep both seasonality rows so selection is auditable.
pooled_validation = pd.DataFrame(validation_rows)
validation_comparison = pd.concat([pooled_validation, pd.DataFrame(ts_validation_rows)], ignore_index=True)
validation_comparison.to_csv(RESULTS_DIR / "validation_seasonality_comparison.csv", index=False)
ts_candidate_df.to_csv(RESULTS_DIR / "arima_order_and_exogenous_selection.csv", index=False)

validation_chosen = validation_comparison.sort_values(["RMSE", "MAE"]).drop_duplicates(
    ["target", "model", "fit_target"]
)
validation_chosen.to_csv(RESULTS_DIR / "validation_model_comparison.csv", index=False)

# Keep unrestricted winners as a diagnostic. The published tentative choice is
# selected among models fitted to the actual project target: one-year growth.
winner_rows = []
all_candidate_winner_rows = []
for target_name in TARGETS:
    all_candidates = validation_chosen[validation_chosen["target"].eq(target_name)]
    eligible = all_candidates[all_candidates["fit_target"].eq(ELIGIBLE_FIT_TARGET)]
    for metric, direction in [("RMSE", "min"), ("MAE", "min"), ("R2", "max")]:
        for subset, output in [(eligible, winner_rows), (all_candidates, all_candidate_winner_rows)]:
            selected = subset.sort_values(metric, ascending=(direction == "min")).iloc[0]
            output.append({"target": target_name, "metric": metric,
                           "winning_model": selected["model"], "fit_target": selected["fit_target"],
                           "quarter_terms": selected["quarter_terms"], "value": selected[metric]})
best_by_metric = pd.DataFrame(winner_rows)
best_by_metric.to_csv(RESULTS_DIR / "validation_metric_winners.csv", index=False)
pd.DataFrame(all_candidate_winner_rows).to_csv(
    RESULTS_DIR / "validation_metric_winners_all_candidates.csv", index=False
)

# Paired within-family comparison prevents a cross-family average from hiding
# the ZHVI exception, where some direct-level fits win in validation.
paired_rows = []
for target_name in TARGETS:
    for family in ["Mean baseline"] + PREDICTOR_MODELS:
        growth_name = display_name(family, GROWTH_FIT)
        level_name = display_name(family, LEVEL_FIT)
        growth = validation_chosen[
            validation_chosen["target"].eq(target_name)
            & validation_chosen["model"].eq(growth_name)
            & validation_chosen["fit_target"].eq(GROWTH_FIT)
        ].iloc[0]
        level = validation_chosen[
            validation_chosen["target"].eq(target_name)
            & validation_chosen["model"].eq(level_name)
            & validation_chosen["fit_target"].eq(LEVEL_FIT)
        ].iloc[0]
        paired_rows.append({
            "target": target_name, "family": family,
            "growth_fit_validation_RMSE": growth["RMSE"],
            "level_fit_validation_RMSE": level["RMSE"],
            "lower_validation_RMSE": "Growth fit" if growth["RMSE"] < level["RMSE"] else "Level fit",
        })
pd.DataFrame(paired_rows).to_csv(RESULTS_DIR / "fit_target_paired_validation_comparison.csv", index=False)


# -----------------------------------------------------------------------------
# 6. Freeze selected configurations, produce validation/test predictions
# -----------------------------------------------------------------------------
train_validation = pd.concat([train, validation], ignore_index=True)
prediction_rows, metric_rows, interval_rows = [], [], []
selected_models = []

for target_name in TARGETS:
    for fit_target in FIT_TARGETS:
        for model_name in ["Mean baseline"] + PREDICTOR_MODELS:
            label = display_name(model_name, fit_target)
            options = validation_comparison[
                validation_comparison["target"].eq(target_name)
                & validation_comparison["model"].eq(label)
                & validation_comparison["fit_target"].eq(fit_target)
            ].sort_values(["RMSE", "MAE"])
            best_option = options.iloc[0]
            include_quarter = best_option["quarter_terms"] == "Yes"
            if model_name == "Mean baseline":
                estimator = LinearRegression()
            else:
                # Reconstruct the selected candidate description from the saved
                # validation rows; direct references are retained in the search.
                estimator = None
                for candidate in REGRESSION_CANDIDATES[model_name]:
                    if fit_description(candidate, fit_target) == best_option["specification"]:
                        estimator = clone(candidate)
                        break
                if estimator is None:
                    # The chosen parameter combination is still present among
                    # candidate rows even if its display label changed.
                    raise RuntimeError(f"Could not restore selected {model_name} candidate")
            selected_models.append((target_name, label, fit_target, include_quarter, estimator,
                                    best_option["specification"]))

            # The validation predictions come from training-only fits used for
            # model/seasonality selection; the test fit may use train+validation.
            for split_name, scoring_data, training_data in [
                ("validation", validation, train), ("test", test, train_validation)
            ]:
                model, predicted_level, _ = fit_predict_pooled(
                    training_data, scoring_data, target_name, fit_target, model_name,
                    clone(estimator), include_quarter,
                )
                actual_level = scoring_data[level_target_column(target_name)].to_numpy()
                metrics = score(actual_level, predicted_level)
                metric_rows.append({
                    "target": target_name, "model": label, "fit_target": fit_target,
                    "split": split_name,
                    "quarter_terms": "Yes" if include_quarter else "No",
                    "specification": best_option["specification"], **metrics,
                })
                prediction_rows.append(scoring_data[[COUNTY_KEY, COUNTY_COLUMN, "year", "quarter"]].assign(
                    target=target_name, model=label, fit_target=fit_target,
                    split=split_name, actual_level=actual_level,
                    prediction_level=predicted_level,
                ))

    # ARIMA/ARIMAX always forecast the original level series.
    for family in ["ARIMA", "ARIMAX"]:
        opts = validation_comparison[
            validation_comparison["target"].eq(target_name)
            & validation_comparison["model"].eq(family)
        ].sort_values(["RMSE", "MAE"])
        best_option = opts.iloc[0]
        include_quarter = best_option["quarter_terms"] == "Yes"
        spec = selected_ts[(target_name, family, include_quarter)]
        for split_name, scoring_data in [("validation", validation), ("test", test)]:
            if CACHE_TIME_SERIES:
                saved = cached_ts_predictions[
                    cached_ts_predictions["target"].eq(target_name)
                    & cached_ts_predictions["model"].eq(family)
                    & cached_ts_predictions["split"].eq(split_name)
                ][[COUNTY_KEY, "year", "quarter", "actual_level", "prediction_level"]]
                forecasts = scoring_data[[COUNTY_KEY, "year", "quarter"]].merge(
                    saved, on=[COUNTY_KEY, "year", "quarter"], how="left", validate="one_to_one"
                )
                assert forecasts["prediction_level"].notna().all()
                assert np.allclose(
                    forecasts["actual_level"],
                    scoring_data[level_target_column(target_name)], rtol=0, atol=1e-5,
                ), "Cached time-series forecasts do not match current inputs"
                predicted = forecasts["prediction_level"].to_numpy()
            else:
                forecasts = forecast_rows(
                    scoring_data, target_name, spec["order"], spec["exogenous"], include_quarter
                )
                predicted = forecasts["prediction"].to_numpy()
            actual = scoring_data[level_target_column(target_name)].to_numpy()
            metric_rows.append({
                "target": target_name, "model": family, "fit_target": LEVEL_FIT,
                "split": split_name, "quarter_terms": "Yes" if include_quarter else "No",
                "specification": f"ARIMA{spec['order']}; {spec['bundle_name']}",
                **score(actual, predicted),
            })
            prediction_rows.append(scoring_data[[COUNTY_KEY, COUNTY_COLUMN, "year", "quarter"]].assign(
                target=target_name, model=family, fit_target=LEVEL_FIT,
                split=split_name, actual_level=actual, prediction_level=predicted,
            ))
            if split_name == "test" and not CACHE_TIME_SERIES:
                interval_rows.append(forecasts.assign(
                    target=target_name, model=family, fit_target=LEVEL_FIT,
                ))

metrics = pd.DataFrame(metric_rows)
predictions = pd.concat(prediction_rows, ignore_index=True)
time_series_intervals = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)


# -----------------------------------------------------------------------------
# 7. Test prediction uncertainty bands from quarter-block bootstrap refits
# -----------------------------------------------------------------------------
NUMBER_OF_BOOTSTRAPS = 100
rng = np.random.default_rng(2026)
periods = train_validation[["year", "quarter"]].drop_duplicates().sort_values(["year", "quarter"])
bootstrap_rows = []
for bootstrap_id in range(NUMBER_OF_BOOTSTRAPS):
    sampled = rng.integers(0, len(periods), size=len(periods))
    blocks = []
    for ix in sampled:
        period = periods.iloc[ix]
        blocks.append(train_validation[
            train_validation["year"].eq(period["year"])
            & train_validation["quarter"].eq(period["quarter"])
        ])
    bootstrap_train = pd.concat(blocks, ignore_index=True)
    for target_name, label, fit_target, include_quarter, estimator, _ in selected_models:
        raw_model = "Mean baseline" if label.startswith("Mean ") else label
        # Baseline rows are represented by their separate visible labels.
        _, forecast, _ = fit_predict_pooled(
            bootstrap_train, test, target_name, fit_target, raw_model,
            clone(estimator), include_quarter,
        )
        for row, value in zip(test.itertuples(index=False), forecast):
            bootstrap_rows.append({
                "target": target_name, "model": label, "fit_target": fit_target,
                COUNTY_KEY: row.county_fips, "year": int(row.year), "quarter": int(row.quarter),
                "bootstrap": bootstrap_id, "prediction_level": float(value),
            })

bootstrap_predictions = pd.DataFrame(bootstrap_rows)
bootstrap_mean = bootstrap_predictions.groupby(
    ["target", "model", "fit_target", "year", "quarter", "bootstrap"], as_index=False
)["prediction_level"].mean()
bootstrap_bands = bootstrap_mean.groupby(
    ["target", "model", "fit_target", "year", "quarter"], as_index=False
)["prediction_level"].agg(
    lower_95=lambda x: x.quantile(0.025), upper_95=lambda x: x.quantile(0.975)
)
if CACHE_TIME_SERIES:
    time_series_bands = cached_ts_level_bands[
        cached_ts_level_bands["model"].isin(["ARIMA", "ARIMAX"])
    ].copy()
    time_series_bands["fit_target"] = LEVEL_FIT
else:
    time_series_bands = time_series_intervals.groupby(
        ["target", "model", "fit_target", "year", "quarter"], as_index=False
    ).agg(lower_95=("lower_95", "mean"), upper_95=("upper_95", "mean"))
confidence_bands = pd.concat([bootstrap_bands, time_series_bands], ignore_index=True)
confidence_bands.to_csv(RESULTS_DIR / "test_confidence_bands.csv", index=False)


# -----------------------------------------------------------------------------
# 8. Validation performance charts and test forecasts on level and growth scales
# -----------------------------------------------------------------------------
validation_chosen = validation_chosen.copy()
validation_chosen["chart_label"] = validation_chosen.apply(
    lambda r: f"{r['model']}\n[{('Growth fit' if r['fit_target'] == GROWTH_FIT else 'Level fit')} | Qtr {r['quarter_terms']}]",
    axis=1,
)
metric_specs = [
    ("RMSE", "Validation RMSE (lower is better)", "validation_rmse_model_comparison.png"),
    ("MAE", "Validation MAE (lower is better)", "validation_mae_model_comparison.png"),
    ("R2", "Validation R² (higher is better)", "validation_r2_model_comparison.png"),
]
for metric, ylabel, filename in metric_specs:
    # Stack targets vertically: their native units and scales differ greatly.
    fig, axes = plt.subplots(2, 1, figsize=(19, 14), sharex=False, sharey=False)
    for ax, target_name in zip(axes, TARGETS):
        rows = validation_chosen[validation_chosen["target"].eq(target_name)].copy()
        rows["order"] = rows["model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
        rows = rows.sort_values(["order", "fit_target"])
        labels = rows["chart_label"].tolist()
        selected = best_by_metric[
            best_by_metric["target"].eq(target_name)
            & best_by_metric["metric"].eq(metric)
        ].iloc[0]
        is_winner = (
            rows["model"].eq(selected["winning_model"])
            & rows["fit_target"].eq(selected["fit_target"])
        ).to_numpy()
        colors = [
            "#d97706" if winner else
            "#5b8fb9" if fit == GROWTH_FIT else
            "#a9b7c6" if model in ["ARIMA", "ARIMAX"] else "#d3dce5"
            for winner, fit, model in zip(is_winner, rows["fit_target"], rows["model"])
        ]
        bars = ax.bar(np.arange(len(rows)), rows[metric], color=colors,
                      edgecolor=["#78350f" if winner else "none" for winner in is_winner],
                      linewidth=1.5)
        values = [f"{value:,.3f}" if metric == "R2" else f"{value:,.0f}" if target_name == "ZHVI"
                  else f"{value:,.1f}" for value in rows[metric]]
        annotations = ax.bar_label(bars, labels=values, padding=3, rotation=90, fontsize=7)
        for label, winner, bar, value_text in zip(annotations, is_winner, bars, values):
            if bar.get_height() < 0:
                # The top of a negative bar is the zero line. Put its value
                # above that line, as with the positive bars.
                label.set_text("")
                ax.annotate(value_text,
                            (bar.get_x() + bar.get_width() / 2, 0),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, fontsize=7)
            if winner:
                label.set_fontweight("bold")
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.6,
                        "*", ha="center", va="center", color="white",
                        fontsize=20, fontweight="bold")
        ax.set_xticks(np.arange(len(rows)), labels, rotation=55, ha="right", fontsize=8)
        unit = "USD" if target_name == "ZHVI" else "USD/month"
        ax.set_title(f"{target_name} ({unit})" if metric != "R2" else target_name)
        ax.set_ylabel(ylabel + (f" [{unit}]" if metric != "R2" else ""))
        ax.grid(axis="y", alpha=0.25)
        if metric == "R2":
            ax.axhline(0, color="black", linewidth=0.8)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom - (top - bottom) * 0.04, top + (top - bottom) * 0.16)
    fig.suptitle(f"Validation model comparison by {metric}", fontsize=15)
    fig.text(0.5, 0.008,
             "Bold * marks the selected growth-fit winner. Blue = growth fit; pale gray = direct level fit; slate = ARIMA family. All bars are scored on the original t+4 Zillow level.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=3.5)
    fig.savefig(RESULTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Summarize winners for RMSE, MAE, and R² (with RMSE as the primary criterion).
print("Validation metric winners (common original-level evaluation scale):")
print(best_by_metric.to_string(index=False))


# Test predictions: preserve the requested order and distinguish fitting targets.
display_order = [
    "Mean growth baseline", "Mean level baseline",
    "Ordinary Linear Regression", "LASSO", "Elastic Net", "Ridge", "XGBoost", "ARIMA", "ARIMAX",
]


def panel_title(target_name, model, fit_target):
    """Show the selected parameters and seasonal setting in each test panel."""
    config = validation_chosen[
        validation_chosen["target"].eq(target_name)
        & validation_chosen["model"].eq(model)
        & validation_chosen["fit_target"].eq(fit_target)
    ].iloc[0]
    if model in ["ARIMA", "ARIMAX"]:
        specification = config["specification"]
        title = f"{model}{specification.split(';')[0].replace('ARIMA', '')}"
        if model == "ARIMAX":
            title += f"; {specification.split('; ')[-1]}"
    else:
        settings = config["specification"].split("; ")[-1]
        title = f"{model} ({settings})"
    short_fit = "growth" if fit_target == GROWTH_FIT else "level"
    return f"{title}\nFit: {short_fit}; quarter terms: {config['quarter_terms']}"


test_predictions = predictions[predictions["split"].eq("test")].copy()
average_actual = test_predictions.drop_duplicates(
    ["target", "county_fips", "year", "quarter"]
).groupby(["target", "year", "quarter"], as_index=False)["actual_level"].mean()

for target_name in TARGETS:
    group = test_predictions[test_predictions["target"].eq(target_name)]
    avg = group.groupby(["model", "fit_target", "year", "quarter"], as_index=False)["prediction_level"].mean()
    avg = avg.merge(confidence_bands[confidence_bands["target"].eq(target_name)],
                    on=["model", "fit_target", "year", "quarter"], how="left")
    avg["period"] = avg["year"].astype(str) + "Q" + avg["quarter"].astype(str)
    actual = average_actual[average_actual["target"].eq(target_name)].copy()
    actual["period"] = actual["year"].astype(str) + "Q" + actual["quarter"].astype(str)
    rows = []
    for model in display_order:
        model_rows = avg[avg["model"].eq(model)]
        if model_rows.empty:
            continue
        for fit_target in model_rows["fit_target"].drop_duplicates():
            rows.append((model, fit_target, model_rows[model_rows["fit_target"].eq(fit_target)]))
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (model, fit_target, forecast) in zip(axes, rows):
        color = plt.cm.tab10(display_order.index(model) % 10)
        ax.plot(actual["period"], actual["actual_level"], color="black", marker="o", label="Actual")
        ax.plot(forecast["period"], forecast["prediction_level"], color=color, marker=".", label="Predicted")
        ax.fill_between(forecast["period"], forecast["lower_95"], forecast["upper_95"], color=color, alpha=0.18, label="95% band")
        ax.set_title(panel_title(target_name, model, fit_target), fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    for ax in axes:
        if ax.get_visible():
            ax.tick_params(axis="x", rotation=45)
    units = "Home value (USD)" if target_name == "ZHVI" else "Rent index (USD/month)"
    fig.suptitle(f"2023–2024 {target_name}: one-year-ahead level forecasts", y=0.995)
    fig.supylabel(f"Average t+4 {units}")
    fig.supxlabel("Prediction origin", y=0.035)
    fig.text(0.5, 0.008, "Regression bands use quarter-block bootstrap refits; ARIMA-family bands are conditional forecast intervals.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.96))
    fig.savefig(RESULTS_DIR / f"{target_name.lower()}_test_prediction_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# Derive the implied one-year growth for every model so growth-fit and level-fit
# approaches can still be compared on the growth outcome scale too.
growth_rows = []
growth_prediction_frames = []
for target_name, info in TARGETS.items():
    subset = test_predictions[test_predictions["target"].eq(target_name)].merge(
        test[[COUNTY_KEY, "year", "quarter", info["level"], info["growth"]]],
        on=[COUNTY_KEY, "year", "quarter"], validate="many_to_one",
    )
    subset["actual_growth_pct"] = subset[info["growth"]]
    subset["predicted_growth_pct"] = 100 * (subset["prediction_level"] / subset[info["level"]] - 1)
    growth_prediction_frames.append(subset[[
        COUNTY_KEY, COUNTY_COLUMN, "year", "quarter", "target", "model", "fit_target",
        "actual_growth_pct", "predicted_growth_pct",
    ]])
    for (model, fit_target), rows in subset.groupby(["model", "fit_target"]):
        config = validation_chosen[
            validation_chosen["target"].eq(target_name)
            & validation_chosen["model"].eq(model)
            & validation_chosen["fit_target"].eq(fit_target)
        ].iloc[0]
        growth_rows.append({
            "target": target_name, "model": model, "fit_target": fit_target,
            "quarter_terms": config["quarter_terms"],
            **score(rows["actual_growth_pct"], rows["predicted_growth_pct"]),
        })
growth_test_metrics = pd.DataFrame(growth_rows)
growth_test_metrics.to_csv(RESULTS_DIR / "one_year_ahead_growth_test_metrics.csv", index=False)
growth_predictions = pd.concat(growth_prediction_frames, ignore_index=True)
growth_predictions.to_csv(RESULTS_DIR / "one_year_ahead_growth_test_predictions.csv", index=False)
validation_chosen[["target", "model", "fit_target", "quarter_terms", "RMSE"]].rename(
    columns={"RMSE": "validation_RMSE_level"}
).to_csv(RESULTS_DIR / "selected_quarter_terms.csv", index=False)

# Transform uncertainty intervals to implied percentage growth. For pooled
# models, first average the nine county forecasts within each bootstrap draw.
bootstrap_growth_frames = []
ts_growth_frames = []
for target_name, info in TARGETS.items():
    origins = test[[COUNTY_KEY, "year", "quarter", info["level"]]]
    pooled = bootstrap_predictions[bootstrap_predictions["target"].eq(target_name)].merge(
        origins, on=[COUNTY_KEY, "year", "quarter"], validate="many_to_one",
    )
    pooled["predicted_growth_pct"] = 100 * (pooled["prediction_level"] / pooled[info["level"]] - 1)
    bootstrap_growth_frames.append(pooled)
    if not CACHE_TIME_SERIES:
        ts = time_series_intervals[time_series_intervals["target"].eq(target_name)].merge(
            origins, on=[COUNTY_KEY, "year", "quarter"], validate="many_to_one",
        )
        ts["lower_growth_pct"] = 100 * (ts["lower_95"] / ts[info["level"]] - 1)
        ts["upper_growth_pct"] = 100 * (ts["upper_95"] / ts[info["level"]] - 1)
        ts_growth_frames.append(ts)

bootstrap_growth = pd.concat(bootstrap_growth_frames, ignore_index=True)
bootstrap_growth_mean = bootstrap_growth.groupby(
    ["target", "model", "fit_target", "year", "quarter", "bootstrap"], as_index=False
)["predicted_growth_pct"].mean()
bootstrap_growth_bands = bootstrap_growth_mean.groupby(
    ["target", "model", "fit_target", "year", "quarter"], as_index=False
)["predicted_growth_pct"].agg(
    lower_95=lambda x: x.quantile(0.025), upper_95=lambda x: x.quantile(0.975)
)
if CACHE_TIME_SERIES:
    ts_growth_bands = cached_ts_growth_bands[
        cached_ts_growth_bands["model"].isin(["ARIMA", "ARIMAX"])
    ].copy()
    ts_growth_bands["fit_target"] = LEVEL_FIT
else:
    ts_growth_bands = pd.concat(ts_growth_frames, ignore_index=True).groupby(
        ["target", "model", "fit_target", "year", "quarter"], as_index=False
    ).agg(lower_95=("lower_growth_pct", "mean"), upper_95=("upper_growth_pct", "mean"))
growth_bands = pd.concat([bootstrap_growth_bands, ts_growth_bands], ignore_index=True)
growth_bands.to_csv(RESULTS_DIR / "one_year_ahead_growth_test_bands.csv", index=False)

for target_name in TARGETS:
    target_rows = growth_predictions[growth_predictions["target"].eq(target_name)]
    actual = target_rows.drop_duplicates([COUNTY_KEY, "year", "quarter"]).groupby(
        ["year", "quarter"], as_index=False
    )["actual_growth_pct"].mean()
    actual["period"] = actual["year"].astype(str) + "Q" + actual["quarter"].astype(str)
    avg = target_rows.groupby(["model", "fit_target", "year", "quarter"], as_index=False)[
        "predicted_growth_pct"
    ].mean().merge(
        growth_bands[growth_bands["target"].eq(target_name)],
        on=["model", "fit_target", "year", "quarter"], how="left", validate="one_to_one",
    )
    avg["period"] = avg["year"].astype(str) + "Q" + avg["quarter"].astype(str)
    rows = []
    for model in display_order:
        model_rows = avg[avg["model"].eq(model)]
        for fit_target in model_rows["fit_target"].drop_duplicates():
            rows.append((model, fit_target, model_rows[model_rows["fit_target"].eq(fit_target)]))
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (model, fit_target, forecast) in zip(axes, rows):
        color = plt.cm.tab10(display_order.index(model) % 10)
        ax.plot(actual["period"], actual["actual_growth_pct"], color="black", marker="o", label="Actual")
        ax.plot(forecast["period"], forecast["predicted_growth_pct"], color=color, marker=".", label="Predicted")
        ax.fill_between(forecast["period"], forecast["lower_95"], forecast["upper_95"], color=color, alpha=0.18, label="95% band")
        ax.set_title(panel_title(target_name, model, fit_target), fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    for ax in axes:
        if ax.get_visible():
            ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"2023–2024 {target_name}: implied one-year-ahead growth", y=0.995)
    fig.supylabel(f"Average {target_name} growth (%)")
    fig.supxlabel("Prediction origin", y=0.035)
    fig.text(0.5, 0.008, "Every level forecast is converted to implied growth at its county's observed origin level.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.96))
    fig.savefig(RESULTS_DIR / f"{target_name.lower()}_1y_ahead_growth_test_prediction_comparison.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 9. Model-agnostic permutation importance for each target's best pooled model
# -----------------------------------------------------------------------------
# Interpret the growth-target winners only. Permutations preserve the panel:
# national predictors move as entire quarter blocks; county predictors are
# shuffled among counties within each quarter. The 36 validation rows make
# these exploratory sensitivity measures, not significance tests.
importance_rows = []
importance_model_rows = []
for target_name in TARGETS:
    candidate = validation_chosen[
        validation_chosen["target"].eq(target_name)
        & validation_chosen["fit_target"].eq(ELIGIBLE_FIT_TARGET)
        & validation_chosen["model"].isin(PREDICTOR_MODELS)
    ].sort_values(["RMSE", "MAE"]).iloc[0]
    label = candidate["model"]
    fit_target = candidate["fit_target"]
    include_quarter = candidate["quarter_terms"] == "Yes"
    specification = candidate["specification"]
    estimator = next(
        clone(e) for e in REGRESSION_CANDIDATES[label]
        if fit_description(e, fit_target) == specification
    )
    numeric_features = COMMON_PREDICTORS.copy()
    if fit_target == LEVEL_FIT:
        numeric_features.append(TARGETS[target_name]["level"])
    columns = numeric_features + [COUNTY_COLUMN] + (["quarter"] if include_quarter else [])
    outcome = TARGETS[target_name]["growth"] if fit_target == GROWTH_FIT else level_target_column(target_name)
    model = make_pooled_pipeline(numeric_features, estimator, include_quarter)
    model.fit(train[columns], train[outcome])
    actual_level = validation[level_target_column(target_name)].to_numpy()

    def level_forecast(input_rows):
        raw = model.predict(input_rows[columns])
        if fit_target == GROWTH_FIT:
            return input_rows[TARGETS[target_name]["level"]].to_numpy() * (1 + raw / 100)
        return raw

    unpermuted_rmse = score(actual_level, level_forecast(validation))["RMSE"]
    for variable_index, variable in enumerate(numeric_features):
        changes = []
        quarter_shared = validation.groupby(["year", "quarter"])[variable].nunique(dropna=False).max() == 1
        for repeat in range(30):
            # Shuffle one feature without inventing different national values
            # for counties in the same quarter. County/quarter controls stay fixed.
            changed = validation.copy()
            random = np.random.default_rng(2026 + 1000 * variable_index + repeat)
            if quarter_shared:
                quarter_values = validation.drop_duplicates(["year", "quarter"])[
                    ["year", "quarter", variable]
                ]
                shuffled = random.permutation(quarter_values[variable].to_numpy())
                value_by_quarter = {
                    (int(y), int(q)): value
                    for y, q, value in zip(quarter_values["year"], quarter_values["quarter"], shuffled)
                }
                changed[variable] = [
                    value_by_quarter[(int(y), int(q))]
                    for y, q in zip(changed["year"], changed["quarter"])
                ]
            else:
                for _, quarter_rows in changed.groupby(["year", "quarter"], sort=False):
                    changed.loc[quarter_rows.index, variable] = random.permutation(
                        quarter_rows[variable].to_numpy()
                    )
            shuffled_rmse = score(actual_level, level_forecast(changed))["RMSE"]
            changes.append(shuffled_rmse - unpermuted_rmse)
        importance_rows.append({
            "target": target_name, "model": label, "fit_target": fit_target,
            "variable": variable,
            "delta_rmse_mean": float(np.mean(changes)),
            "delta_rmse_sd": float(np.std(changes, ddof=1)),
            "delta_rmse_pct_mean": float(100 * np.mean(changes) / unpermuted_rmse),
            "delta_rmse_pct_sd": float(100 * np.std(changes, ddof=1) / unpermuted_rmse),
            "permutation_rule": "whole quarter blocks" if quarter_shared else "across counties within quarter",
            "unpermuted_validation_rmse": unpermuted_rmse,
        })
    importance_model_rows.append({
        "target": target_name, "model": label, "fit_target": fit_target,
        "quarter_terms": candidate["quarter_terms"],
        "validation_RMSE_level": candidate["RMSE"],
        "explanation": "Growth-fitted winner by validation level RMSE; 30 panel-preserving permutations per feature on 36 validation rows.",
    })

importance = pd.DataFrame(importance_rows)
importance.to_csv(RESULTS_DIR / "winning_model_feature_importance.csv", index=False)
pd.DataFrame(importance_model_rows).to_csv(RESULTS_DIR / "winning_model_feature_importance_notes.csv", index=False)
for target_name in TARGETS:
    rows = importance[importance["target"].eq(target_name)].sort_values("delta_rmse_pct_mean")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    top_three = set(rows.nlargest(3, "delta_rmse_pct_mean")["variable"])
    colors = ["#1d4ed8" if variable in top_three and value > 0 else
              "#93c5fd" if value > 0 else "#9ca3af"
              for variable, value in zip(rows["variable"], rows["delta_rmse_pct_mean"])]
    ax.barh(rows["variable"], rows["delta_rmse_pct_mean"],
            xerr=rows["delta_rmse_pct_sd"], color=colors, capsize=2)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Change in validation level RMSE after permutation (%)")
    ax.grid(axis="x", alpha=0.25)
    selected_note = next(item for item in importance_model_rows if item["target"] == target_name)
    ax.set_title(f"{target_name}: {selected_note['model']} fitted to one-year growth\n"
                 "Validation permutation sensitivity (36 county-quarters)")
    fig.text(0.5, 0.01,
             "Whiskers = SD across 30 shuffles, not confidence intervals. Negative values mean shuffling improved RMSE. County/quarter controls are excluded.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS_DIR / f"{target_name.lower()}_winning_model_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 10. Human-readable result summary for notebook output
# -----------------------------------------------------------------------------
print("Validation winners, by target and metric (primary choice: lowest RMSE):")
print(best_by_metric.to_string(index=False))
print("\nFeature-based pooled winners used for interpretation:")
print(pd.DataFrame(importance_model_rows).to_string(index=False))
print(f"\nAll tables and figures saved under: {RESULTS_DIR}")

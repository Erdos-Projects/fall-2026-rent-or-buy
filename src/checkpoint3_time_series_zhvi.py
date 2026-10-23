"""Checkpoint 3, Person 3: exploratory time-series models for 1-year-ahead ZHVI growth.

Compares ARIMA and ARIMAX with the Linear and Ridge regression baselines on the shared
chronological split, using MAE, RMSE and R^2.

Run from the repository root:
    python src/checkpoint3_time_series_zhvi.py

Inputs (read only)
    data/processed/1-year-ahead split/{train,validation,test}.csv
    data/processed/feature engineered data/master_county_quarter_features.csv
        (ZHVI and mortgage-rate history up to each prediction origin, for the time-series models)

Outputs (results/checkpoint3_time_series/)
    metrics.csv         MAE, RMSE, R^2 per model on validation and test
    predictions.csv     one row per county x prediction origin x model
    test_by_quarter.png test predictions vs actual, 9-county average

Protocol (docs/checkpoint3_chronological_splitting.md)
    * Ridge alpha and the ARIMA/ARIMAX order are chosen on validation only; test is scored once.
    * Regression models: fit on train, choose on validation, refit on train + validation, score test.
    * Time-series models: rolling origin. For each county and prediction origin t, fit on that
      county's history from 2015Q1 through t only, then forecast 4 quarters ahead.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "1-year-ahead split"
PANEL = ROOT / "data" / "processed" / "feature engineered data" / "master_county_quarter_features.csv"
OUT = ROOT / "results" / "checkpoint3_time_series"

TARGET = "zhvi_growth_ahead_1y_pct"
TARGETS = ["zhvi_growth_ahead_1y_pct", "zori_growth_ahead_1y_pct"]
KEYS = ["county_fips", "county_name", "year", "quarter"]
HORIZON = 4  # quarters ahead

RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
# Candidate orders for log ZHVI with one difference (= quarterly growth) plus drift.
ARIMA_ORDERS = [(0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 1, 1), (1, 1, 1)]
# ARIMAX exogenous variable. Its future is unknown at time t, so it is held at its last value.
EXOG = "nat_mortgage_rate_30y_pct"

NON_CONVERGED: list[str] = []


def metrics(y_true, y_pred) -> dict:
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                R2=r2_score(y_true, y_pred))


# ----------------------------------------------------------------------------- regression baselines

def regression_baselines(train, val, test):
    """Linear and Ridge regression on all numeric features (both one-year targets excluded)."""
    drop = set(TARGETS) | {"county_fips", "county_name", "year", "quarter_end_date"}
    X = [c for c in train.columns if c not in drop and pd.api.types.is_numeric_dtype(train[c])]
    train_val = pd.concat([train, val], ignore_index=True)

    def pipe(model):
        # Imputer and scaler are fit inside the pipeline, i.e. on training data only.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    val_rmse = {a: metrics(val[TARGET], pipe(Ridge(alpha=a)).fit(train[X], train[TARGET]).predict(val[X]))["RMSE"]
                for a in RIDGE_ALPHAS}
    alpha = min(val_rmse, key=val_rmse.get)
    print(f"Ridge validation RMSE by alpha: { {a: round(v, 2) for a, v in val_rmse.items()} } -> alpha={alpha}")

    out = {}
    for name, model in (("Linear regression", LinearRegression()), (f"Ridge (alpha={alpha})", Ridge(alpha=alpha))):
        out[name] = (pipe(model).fit(train[X], train[TARGET]).predict(val[X]),
                     pipe(model).fit(train_val[X], train_val[TARGET]).predict(test[X]))
    return out


# ----------------------------------------------------------------------------- time-series models

def county_histories() -> dict[str, pd.DataFrame]:
    """Per county: 100 x log ZHVI and the mortgage rate, indexed by quarter.

    The x100 scale is for numerical stability: on the raw log scale the quarterly changes are
    tiny and the likelihood optimizer often fails to converge. It does not change the model.
    """
    panel = pd.read_csv(PANEL)
    panel["period"] = [pd.Period(year=int(y), quarter=int(q), freq="Q") for y, q in zip(panel["year"], panel["quarter"])]
    out = {}
    for fips, g in panel.groupby("county_fips"):
        g = g.sort_values("period").set_index("period")
        assert len(g) == 44 and (g["zhvi_usd"] > 0).all()
        out[fips] = pd.DataFrame({"y": 100 * np.log(g["zhvi_usd"]), EXOG: g[EXOG]})
    return out


def forecast_growth(hist: pd.DataFrame, origin: pd.Period, order, use_exog: bool) -> float:
    """Fit on data through `origin` only; return the forecast 1-year growth in percent."""
    h = hist.loc[:origin]
    exog = h[[EXOG]] if use_exog else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = ARIMA(h["y"], exog=exog, order=order, trend="t").fit()
        future_exog = np.repeat(h[[EXOG]].values[-1:], HORIZON, axis=0) if use_exog else None
        fc = fit.forecast(steps=HORIZON, exog=future_exog)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        NON_CONVERGED.append(f"ARIMA{order} exog={use_exog} at {origin}")
    return 100 * (np.exp((fc.iloc[-1] - h["y"].iloc[-1]) / 100) - 1)


def time_series_models(val, test):
    hist = county_histories()

    def predict(df, order, use_exog):
        return np.array([forecast_growth(hist[r.county_fips], pd.Period(year=r.year, quarter=r.quarter, freq="Q"),
                                         order, use_exog) for r in df.itertuples(index=False)])

    out = {}
    for family, use_exog in (("ARIMA", False), ("ARIMAX (mortgage rate)", True)):
        val_pred = {o: predict(val, o, use_exog) for o in ARIMA_ORDERS}
        val_rmse = {o: metrics(val[TARGET], p)["RMSE"] for o, p in val_pred.items()}
        best = min(val_rmse, key=val_rmse.get)
        print(f"{family} validation RMSE by order: { {str(o): round(v, 2) for o, v in val_rmse.items()} } -> {best}")
        out[f"{family} {best}"] = (val_pred[best], predict(test, best, use_exog))
    return out


# ----------------------------------------------------------------------------- report

def plot_test(test: pd.DataFrame, models: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin = test["year"].astype(str) + "Q" + test["quarter"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    actual = test[TARGET].groupby(origin).mean()
    ax.plot(actual.index, actual.values, color="black", lw=2.5, marker="o", label="Actual")
    for name, (_, test_pred) in models.items():
        s = pd.Series(test_pred, index=test.index).groupby(origin).mean()
        ax.plot(s.index, s.values, marker=".", lw=1.5, label=name)
    ax.set_ylabel("1-year-ahead ZHVI growth (%)")
    ax.set_xlabel("Prediction origin")
    ax.set_title("Test period: predicted vs actual (average of 9 counties)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = (pd.read_csv(SPLIT_DIR / f"{s}.csv") for s in ("train", "validation", "test"))

    models = {"Mean baseline": (np.full(len(val), train[TARGET].mean()),
                                np.full(len(test), pd.concat([train, val])[TARGET].mean()))}
    models.update(regression_baselines(train, val, test))
    models.update(time_series_models(val, test))

    rows, preds = [], []
    for name, (val_pred, test_pred) in models.items():
        for split, df, p in (("validation", val, val_pred), ("test", test, test_pred)):
            rows.append(dict(model=name, split=split, **metrics(df[TARGET], p), n=len(df)))
            preds.append(df[KEYS + [TARGET]].assign(model=name, split=split, prediction=p))
    metrics_df = pd.DataFrame(rows)
    metrics_df.round(4).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(preds).round(4).to_csv(OUT / "predictions.csv", index=False)
    plot_test(test, models, OUT / "test_by_quarter.png")

    for split in ("validation", "test"):
        print(f"\n=== {split} ===")
        print(metrics_df[metrics_df["split"] == split].drop(columns="split").round(3).to_string(index=False))
    print(f"\nTime-series fits that did not converge: {len(NON_CONVERGED)}")
    print(f"Saved results to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

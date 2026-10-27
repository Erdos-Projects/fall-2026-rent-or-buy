# Checkpoint 2: Feature Engineering and Quality Control

Run the reproducible pipeline from the repository root:

```bash
python src/build_master_county_quarter.py
python src/build_master_county_quarter_features.py
```

## Outputs

- `data/processed/merged data/master_county_quarter.csv`: cleaned source panel, ordered as keys, housing, demographics, national economics, county labor, Ohio labor, and source-quality flags.
- `data/processed/feature engineered data/master_county_quarter_features.csv`: model-ready panel. It retains the source groups, then places engineered historical and affordability features, followed by two one-year-ahead prediction labels.
- `data/processed/feature engineered data/master_county_quarter_features_missingness.csv`: missingness, affected counties, and affected periods for every feature-panel column.
- `data/processed/feature engineered data/master_county_quarter_features_quality_summary.csv`: descriptive statistics for engineered features and labels.
- `data/metadata/master_feature_dictionary.csv`: one definition for every feature-panel column.

Both panels contain 396 rows: nine counties and 44 quarterly observations from 2015Q1 through 2025Q4. Keys are unique.

## Feature and Target Rules

For county $c$ and quarter $t$, backward-looking housing growth uses only observations at or before $t$:

- quarter-over-quarter: $100\left(X_t / X_{t-1} - 1\right)$
- previous one year: $100\left(X_t / X_{t-4} - 1\right)$

Rate changes use percentage points. For example, if $r_t$ is the mortgage rate, the one-year change is $r_t - r_{t-4}$.

The two labels are intentionally forward-looking and must never be model inputs:

$$
g_{t,4} = 100\left(\frac{X_{t+4}}{X_t} - 1\right).
$$

This produces 360 observed labels for each one-year target. The final four quarters of each county lack labels because the observed panel ends at 2025Q4.

The project discontinued the three-year prediction task because the available
2015Q1–2025Q4 panel cannot support a useful leakage-free train/validation/test
split at that horizon. Therefore, the feature script does not create
`zhvi_growth_prev_3y_pct`, `zori_growth_prev_3y_pct`,
`zhvi_growth_ahead_3y_pct`, or `zori_growth_ahead_3y_pct`.

## Important Limitations

- ACS annual estimates are repeated within their reference-year quarters. Values remain missing for 2020 and 2025, and no values are imputed. ACS features must be lagged to their public release date in a real-time modeling design.
- `nat_treasury_1y_pct` is the quarterly mean FRED DGS1 yield. The feature
  `treasury_1y_change_prev_1y_pp` equals the current quarterly yield minus
  the yield four quarters earlier. It is a backward-looking change in
  percentage points and has no observed value in each county's first four
  quarters.
- CPI and BLS labor measures in 2025Q4 are based on November–December observations; the retained `*_partial*` flags identify those nine rows.
- The feature script rejects duplicate keys, unexpected counties, incomplete county histories, infinite values, and non-positive affordability ratios. It does not silently remove suspicious values.

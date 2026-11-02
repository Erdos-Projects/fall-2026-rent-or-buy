# Project documentation

New to the project? Start with the [main README](../README.md), then follow the checkpoints below. Source notes explain where the data came from; the [data dictionaries](../data/metadata/) define the final variables.

| Topic | Document | What it explains |
|---|---|---|
| Data sources | [ACS demographics](acs-demographics-data.md) | Census variables, coverage, and processing |
| Data sources | [Economy and labor](econ_labor_data.md) | FRED and BLS measures, units, and coverage |
| Checkpoint 1 | [Data integration](checkpoint1_data_integration.md) | Cleaning, county keys, and the merged county-quarter panel |
| Checkpoint 2 | [Feature engineering](checkpoint2_feature_engineering.md) | Historical predictors, one-year targets, and quality checks |
| Checkpoint 3 | [Chronological splitting](checkpoint3_chronological_splitting.md) | Train/validation/test origins and leakage controls |
| Checkpoint 3 | [Evaluation plan](evaluation_plan.md) | Metrics, baselines, and evaluation rules |
| Checkpoint 3 | [Time-series exploration](checkpoint3_time_series.md) | ARIMA and ARIMAX experiments |
| Checkpoint 4 | [Model comparison](checkpoint4_model_compare.md) | Validation selection, test caveats, and feature sensitivity |

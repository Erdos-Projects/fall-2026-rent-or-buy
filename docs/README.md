# Project Documentation

This folder contains durable project documentation that should be reviewed alongside code and data changes.

## Documents
- `acs-demographics-data.md` - information regarding the measures derived from the American Community Survey and how they were processed for the project
- `econ-labor-data.md` - information regarding the national- and state-level economic and labor measures derived from FRED and BLS and how they were processed for the project
- `checkpoint1_data_integration.md` - documentation of how all collected data sources and measures were cleaned, integrated, and merged to create the master county-quarter dataset
- `checkpoint2_feature_engineering.md` - documentaiton of how the master dataset was transformed into a model-ready dataset through feature engineering, the creation of historical measures and one-year-ahead prediction targets, and data quality checks
- `checkpoint3_chronological_splitting.md` - documentation of how model-ready dataset was divided into chronological training, validation and test sets with controls to prevent data leakage and allow for an unbiased evaluation
- `checkpoint3_evaluation_plan.md` - documentation of the model evaluation framework and comparison of model predictive performance
- `checkpoint3_time_series.md` - documentation of exploratory time-series modeling (ARIMA and ARIMAX) to see whether time-series appraoches improve ZHVI growht predictions compared with linear and ridge regression baselines
- `checkpoint4_model_compare.md` - documentation of comparison of 8 modeling approaches, including validation-based model selection and performance evaluation
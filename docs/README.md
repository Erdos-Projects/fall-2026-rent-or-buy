# Project Documentation

This folder contains durable project documentation that should be reviewed alongside code and data changes.

## Suggested documents

- `checkpoint1.md` — problem definition, scope, data assessment, learnability assessment, and KPI definitions.
- `data_dictionary.md` — variable names, descriptions, units, sources, geography, and annualization rules.
- `data_sources.md` — download URLs, source versions, coverage, licenses, and known limitations.
- `decisions.md` — important project decisions and the reasons for them.
- `changes.md` — a dated record of major data, code, and documentation changes.

## Documentation rules

Document decisions that affect the analysis, especially geography definitions, boundary vintages, missing-data handling, annualization methods, and source revisions. Link each change to the related GitHub Issue or Pull Request when possible.

Use clear dates in `YYYY-MM-DD` format. Do not store API keys, credentials, or unmodified third-party data in this folder.

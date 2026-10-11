## Demographic Data Sources

### Source and Geographic Scope
The `acs_{year}_all_counties.csv` files were downloaded from the U.S. Census
ACS 1-year API. They contain a wider Ohio county extract in their raw form.
The integration pipeline retains only the project's nine target counties using
five-digit county FIPS codes: Butler, Cuyahoga, Franklin, Hamilton, Lucas,
Montgomery, Stark, Summit, and Warren.

### Time Coverage and Missingness
The available source years are 2015–2019 and 2021–2024. ACS 1-year estimates
were not released for 2020, and 2025 estimates were not available when the
files were collected. In the county-quarter panel, all four quarters of 2020
and 2025 remain missing. No ACS value is carried forward or imputed.

### Variables Downloaded
For each county, the following estimates and margins of error were acquired:
- B01003: Total Population
- B19013: Median Household Income in the Past 12 Months (in Inflation-Adjusted Dollars)
- B25001: Number of Housing Units
- B25002_003: Occupancy Status - Number of Vacancies
- B25002_001: Occupancy Status - Total

### Processed Variables
The integration pipeline writes `data/processed/ACS/acs_county_year.csv` with
standardized names. `acs_county_quarter.csv` repeats each observed annual value
across Q1–Q4 of that same year. The processed variables include:

- `population` and `population_moe`
- `median_household_income_usd` and `median_household_income_moe_usd`
- `housing_units` and `housing_units_moe`
- `vacant_housing_units` and `vacant_housing_units_moe`
- `occupancy_total_units` and `occupancy_total_units_moe`
- `vacancy_rate_pct = 100 × vacant_housing_units / occupancy_total_units`

The Census missing-value sentinel `-555555555` is converted to a missing value.

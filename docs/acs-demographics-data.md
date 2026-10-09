## Demographic Data Sources

### Data Source
The acs*.csv were downloaded from the US census (https://www.census.gov/programs-surveys/acs/data.html). The data were downloaded using the Census API with an API key. 

### Data Range & Missingness
Note that 2020 and 2025 are missing. There was no data available for 2020, and 2025 data has not yet been released (as of 9/30/2026).

### Metropolitan Statistical Areas (MSA)
The following MSAs were considered:
- Cincinnati, OH-KY-IN Metro Area
- Cleveland-Elyria, OH Metro Area
- Cleveland, OH Metro Area
- Columbus, OH Metro Area
- Akron, OH Metro Area
- Toledo, OH Metro Area

For Cleveland, the metro area was called Cleveland-Elyria until 2022. Beginning 2023, it became Cleveland only and MSA code changed.

### Data Downloaded
For each MSA, the following estimates and measures of error were acquired:
- B01003: Total Population
- B19013: Median Household Income in the Past 12 Months (in Inflation-Adjusted Dollars)
- B25001: Number of Housing Units
- B25002_003: Occupancy Status - Number of Vacancies
- B25002_001: Occupancy Status - Total

### Variables
In the all_acs.csv file, the following measures are included for each MSA x year:
- total_population_estimate: estimated total population
- total_population_MoE: margin error of total population
- median_HH_income_estimate: estimate of the median household income
- median_HH_income_MoE: margin of error of the median household income
- total_housing_units_estimate: estimate total number of housing units
- total_housing_units_MoE: margin of error of the total number of housing units
- vacant_housing_units_estimate: estimate number of vacant housing units
- vacant_housing_units_MoE: margin of error of vacant housing units
- vacancy_rate_pct: percent of vacant units out of all housing units (100* (vacant_housing_units_estimate / total_housing_units_estimate))
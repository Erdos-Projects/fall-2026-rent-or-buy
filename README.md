# fall-2026-rent-or-buy
Team project: fall-2026-rent-or-buy
Team members: Zixuan (Shawn) Chen, Callie Reid, Toma Debnath, Da Eun Jung, and Youngho Lee


## Project Overview
This project examines whether current housing, demographic, and economic conditions can predict one-year-ahead Zillow Home Value Index (ZHVI) and Zillow Observed Rent Index (ZORI) growth across nine Ohio counties.

The resulting growth predictions are used in a downstream Rent-versus-Buy Financial Calculator to estimate the future financial advantage of buying a home versus renting and investing available cash in a low-risk alternative.

## Project Scope
- Geographic Coverage: Butler, Cuyahoga, Franklin, Hamilton, Lucas, Montgomery, Stark, Summit, Warren
- Time Period: 2015 Q1 - 2025 Q4
- Primary Outcomes: 1-year-ahead ZHVI and ZORI growth
- Data Frequency: quarterly
- Data Sources: Zillow, American Community Survey (ACS), Federal Housing Finance Agency (FHFA), Federal Reserve Economic Data (FRED), and Bureau of Labor Statistics (BLS)

## Project Workflow
1. Data Integration: collect, clean, and integrate housing, demographic, and economic data into a county-quarter dataset
2. Feature Engineering: create historical and derived predictors, construct future hosuing growth targets, and conduct data quality checks
3. Data Splitting and Evaluation: establish chronological training, validation, and test sets while accounting for temporal ordering and preventing any data leakage
4. Model Development and Comparison: estimate and compare multiple predictive modeling approaches

## Repository Structure
- data/ - raw and processed datasets
- docs/ - documentation of data sources and "checkpoints" throughout the project
- notebooks/ - contains a Jupyter notebook providing a walkthrough of the one-year-ahead ZHVI and ZORI model comparisons
- results/ - model outputs, evaluation results, and related project artifacts
- src/ - Python scrips to clean, integrate, transform, and analyze data

## Final Project Product
View the Rent-versus-Buy Financial Calculator in the repository (./index.html)


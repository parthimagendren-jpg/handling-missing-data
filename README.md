# Handling Missing Data: Mean and Regression Imputation

Comparison of two imputation methods, mean imputation and linear regression imputation, on a housing dataset, with before/after visualisations (heatmaps and histograms).

Coursework for TPD3353 Introduction to Data Analytics.

## Dataset

Sberbank Russian Housing Market data (Kaggle): https://www.kaggle.com/c/sberbank-russian-housing-market/data

Download `sberbank.csv` and place it in a `data/` folder:

```
data/sberbank.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python mean_imputation.py        # mean imputation, shows plots
python regression_imputation.py  # regression imputation, shows plots and saves sberbank_regression_cleaned.csv
```

## Method summary

- **Mean imputation:** numeric missing values are replaced with the column mean; non-numeric with `"unknown"`. Duplicates are dropped.
- **Regression imputation:** each numeric column with missing values is predicted by a `LinearRegression` trained on the rows where it is present, using the other numeric columns (missing predictors are filled with their means). Non-numeric columns are filled with the mode.

## Findings

- Mean imputation is simple and fast, but it creates a spike at the mean and shrinks variance.
- Regression imputation preserves relationships between variables and keeps the distribution closer to the original.

## Repository layout

```
figures/   plots from the analysis (before/after heatmaps and histograms)
report/    written report (HANDLING_MISSING_DATA.docx)
mean_imputation.py
regression_imputation.py
requirements.txt
```

## Author

Magendren A/L Parthiban (Matric 2240312)

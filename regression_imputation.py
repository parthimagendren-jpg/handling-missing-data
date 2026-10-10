"""Regression imputation of missing numeric values on the Sberbank housing dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

DATA_PATH = Path("data/sberbank.csv")
OUTPUT_PATH = Path("sberbank_regression_cleaned.csv")
MISSING_VALUES = ["n/a", "na", "--", "NA", "?", ""]

df = pd.read_csv(DATA_PATH, na_values=MISSING_VALUES)

print("===== ORIGINAL DATA =====")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

df = df.drop(columns=["id", "timestamp"], errors="ignore")

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap=["blue", "yellow"])
plt.title("Missing Data (Before Regression Imputation)")
plt.show()

if "life_sq" in df.columns:
    plt.figure()
    plt.hist(df["life_sq"].dropna(), bins=30)
    plt.title("Histogram of life_sq (BEFORE REGRESSION)")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()

df.drop_duplicates(inplace=True)


def regression_impute(df, target_col):
    """Fill missing values of target_col with a linear model on the other numeric columns."""
    df_not_null = df[df[target_col].notnull()]
    df_null = df[df[target_col].isnull()]

    if df_null.shape[0] == 0:
        return df

    df_not_null_num = df_not_null.select_dtypes(include=np.number)
    df_null_num = df_null.select_dtypes(include=np.number)

    if target_col not in df_not_null_num.columns:
        return df

    X_train = df_not_null_num.drop(columns=[target_col])
    y_train = df_not_null_num[target_col]
    X_test = df_null_num.drop(columns=[target_col])

    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    valid_cols = X_train.columns[
        (X_train.isnull().sum() == 0) & (X_test.isnull().sum() == 0)
    ]
    X_train = X_train[valid_cols]
    X_test = X_test[valid_cols]

    if X_train.shape[1] == 0:
        print(f"Skipped: {target_col}")
        return df

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    df.loc[df[target_col].isnull(), target_col] = predicted
    print(f"Filled: {target_col}")
    return df


df_numeric = df.select_dtypes(include=np.number)
for col in df_numeric.columns:
    if df[col].isnull().sum() > 0:
        df = regression_impute(df, col)

# Mode (or "Unknown") for non-numeric columns
df_non_numeric = df.select_dtypes(exclude=np.number)
for col in df_non_numeric.columns:
    if df[col].isnull().sum() > 0:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

print("\nRemaining missing values:", df.isnull().sum().sum())

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap=["blue", "yellow"])
plt.title("Missing Data (After Regression Imputation)")
plt.show()

if "life_sq" in df.columns:
    plt.figure()
    plt.hist(df["life_sq"], bins=30)
    plt.title("Histogram of life_sq (AFTER REGRESSION IMPUTATION)")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned dataset saved as {OUTPUT_PATH}")

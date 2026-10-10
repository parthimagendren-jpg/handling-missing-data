"""Mean imputation of missing values on the Sberbank housing dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/sberbank.csv")
MISSING_VALUES = ["n/a", "na", "--"]

df = pd.read_csv(DATA_PATH, na_values=MISSING_VALUES)

total_cells = np.prod(df.shape)
missing_before = df.isnull().sum().sum()
pct_missing_before = (missing_before / total_cells) * 100

print("--- BEFORE CLEANING ---")
print(f"Total Missing Values: {missing_before}")
print(f"Percentage of Missing Data: {pct_missing_before:.2f}%\n")

plt.style.use("ggplot")
plt.figure(figsize=(12, 5))
cols = df.columns[:30]
colours = ["#000099", "#ffff00"]
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours), cbar=False)
plt.title("1. Heatmap of Missing Data (BEFORE CLEANING)")
plt.show()

col_to_visualize = "life_sq"
plt.figure(figsize=(8, 5))
sns.histplot(df[col_to_visualize].dropna(), kde=True, color="red", bins=30)
plt.title(f"2. Histogram of {col_to_visualize} (BEFORE CLEANING)")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# Mean imputation for numeric columns
df_clean = df.copy()
df_numeric = df_clean.select_dtypes(include=[np.number])
for col in df_numeric.columns:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# "unknown" for non-numeric columns
df_non_numeric = df_clean.select_dtypes(exclude=[np.number])
for col in df_non_numeric.columns:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna("unknown")

df_clean.drop_duplicates(inplace=True)

total_cells_after = np.prod(df_clean.shape)
missing_after = df_clean.isnull().sum().sum()
pct_missing_after = (missing_after / total_cells_after) * 100

print("--- AFTER CLEANING ---")
print(f"Total Missing Values: {missing_after}")
print(f"Percentage of Missing Data: {pct_missing_after:.2f}%\n")

plt.figure(figsize=(12, 5))
sns.heatmap(df_clean[cols].isnull(), cmap=sns.color_palette(colours), cbar=False)
plt.title("3. Heatmap of Missing Data (AFTER CLEANING)")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df_clean[col_to_visualize], kde=True, color="green", bins=30)
plt.title(f"4. Histogram of {col_to_visualize} (AFTER MEAN IMPUTATION)")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/content/archive.zip")

print("First 10 rows:")
print(data.head(10))
print("\nInfo:")
print(data.info())
print("\nAvailable Columns:")
print(data.columns)

df = data.copy()

df.drop(columns=['Dimensions', 'Plot Area'], inplace=True, errors='ignore')

price_col = None
for col in df.columns:
    if "price" in col.lower():
        price_col = col
        break

if price_col is None:
    raise ValueError("No price column found in dataset!")

print(f"\nDetected Price Column: {price_col}")

df[price_col].fillna(df[price_col].median(), inplace=True)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after filling:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")
df.drop_duplicates(inplace=True)
print(f"Data shape after removing duplicates: {df.shape}")

print("\nNull values before dropping:")
print(df.isnull().sum())
df.dropna(inplace=True)
print(f"Shape after dropping nulls: {df.shape}")
print("Null values after dropping:")
print(df.isnull().sum())

Q1 = df[price_col].quantile(0.25)
Q3 = df[price_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df[price_col] >= lower_bound) & (df[price_col] <= upper_bound)]
print(f"\nData shape after removing outliers: {df.shape}")

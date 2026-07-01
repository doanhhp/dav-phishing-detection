import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import joblib

print("========================================")
print("1. Opening the .parquet Dataset (Text)")
print("========================================")
# Load the parquet file
df = pd.read_parquet('data/processed/standardized/main_dataset.parquet')
print(f"Total websites in dataset: {len(df)}")
print("\nFirst 3 rows of the Dataset:")
print(df[['Data', 'Category']].head(3))

print("\n========================================")
print("2. Opening the .joblib Features (Math)")
print("========================================")
# Load the extracted features, labels, and the processor (which holds the feature names)
X_train = joblib.load('data/processed/structural_xgb/X_train.joblib')
y_train = joblib.load('data/processed/structural_xgb/y_train.joblib')
processor = joblib.load('data/processed/structural_xgb/processor.joblib')

# Try to extract feature names from the processor
feature_names = []
try:
    feature_names = processor.get_feature_names()
except AttributeError:
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

print(f"Matrix Shape (Rows, Columns): {X_train.shape}")
print(f"Labels Shape: {y_train.shape}")

print(f"\nFirst row of extracted features (Label = {y_train[0]}):")
# Zip the feature names with the values for the first 15 features
for name, value in zip(feature_names[:15], X_train[0][:15]):
    print(f"  - {name:<35}: {value:.6f}")
print("  ... (truncated for display)")

import pandas as pd
import os

df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
df_html = pd.read_excel("data/raw/OOD_html.xlsx")

df_url['html'] = df_html['Data']
df_url['Label'] = df_url['Category'].map({'ham': 0, 'spam': 1})

df_url.to_parquet("data/processed/OOD_dataset.parquet")
print("Saved OOD dataset to data/processed/OOD_dataset.parquet")

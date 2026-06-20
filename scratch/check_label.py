import pandas as pd
import requests

url = "https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/train-000.parquet"
print(f"Downloading {url}...")
resp = requests.get(url)
with open("test.parquet", "wb") as f:
    f.write(resp.content)

df = pd.read_parquet("test.parquet")
print("Columns:", df.columns)
print("First row label:", df['label'].iloc[0], type(df['label'].iloc[0]))
print("First row target:", df['target'].iloc[0], type(df['target'].iloc[0]))
print("Unique labels:", df['label'].unique())
print("Unique targets:", df['target'].unique())

from huggingface_hub import list_repo_files

files = list_repo_files("phreshphish/phreshphish", repo_type="dataset")
parquet_files = [f for f in files if f.endswith(".parquet") and "train" in f]
print("Parquet files count:", len(parquet_files))
print("Parquet files:", parquet_files[:10])
print("Parquet files end:", parquet_files[-10:])

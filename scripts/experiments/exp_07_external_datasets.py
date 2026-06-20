import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import joblib

# Set paths
EXTERNAL_DIR = "data/raw/external"
os.makedirs(EXTERNAL_DIR, exist_ok=True)

# 1. Download and Process PhreshPhish
print("--- Step 1: Downloading PhreshPhish Dataset ---")
phresh_path = os.path.join(EXTERNAL_DIR, "phreshphish.parquet") # Changed to parquet to avoid CSV null-byte OSError

phresh_data = []
try:
    import requests
    import pyarrow.parquet as pq
    import pyarrow as pa
    import gc
    
    def fetch_and_parse_parquet(url, label_target, max_records, current_count):
        print(f"Downloading {url}...")
        local_path = os.path.join(EXTERNAL_DIR, "temp.parquet")
        try:
            resp = requests.get(url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Streaming {local_path} in safe memory chunks...")
            pf = pq.ParquetFile(local_path)
            # Small batch size prevents 2GB PyArrow C++ memory crashes
            for batch in pf.iter_batches(batch_size=50): 
                df_chunk = batch.to_pandas()
                for i, row in df_chunk.iterrows():
                    label = row.get("label")
                    
                    # Check for Ham
                    if label_target == 0 and label != "benign": continue
                    # Check for Spam
                    if label_target == 1 and label != "phish": continue
                    
                    if current_count >= max_records: break
                    
                    url_str = row.get("url", "")
                    html = row.get("html", "")
                    if isinstance(html, str): html = html[:50000]
                    else: html = ""
                    
                    if not url_str or not html: continue
                    
                    phresh_data.append({"Data": url_str, "html": html, "Category": 1 if label_target == 1 else 0})
                    current_count += 1
                    
                if current_count >= max_records:
                    break
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            
        # Cleanup
        try: os.remove(local_path)
        except: pass
        return current_count

    # Fetch 2500 Ham
    ham_count = 0
    for idx in range(10):
        if ham_count >= 2500: break
        ham_url = f"https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/train-{idx:03d}.parquet"
        ham_count = fetch_and_parse_parquet(ham_url, label_target=0, max_records=2500, current_count=ham_count)
        print(f"Ham Count: {ham_count}/2500")
        
    # Fetch 2500 Spam
    spam_count = 0
    for idx in range(55, 45, -1):
        if spam_count >= 2500: break
        spam_url = f"https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/train-{idx:03d}.parquet"
        spam_count = fetch_and_parse_parquet(spam_url, label_target=1, max_records=2500, current_count=spam_count)
        print(f"Spam Count: {spam_count}/2500")
        
    df_phresh = pd.DataFrame(phresh_data)
    df_phresh = df_phresh.sample(frac=1).reset_index(drop=True) # Shuffle
    df_phresh.to_parquet(phresh_path, index=False)
    print(f"Successfully saved {len(df_phresh)} records to {phresh_path}")
    del df_phresh
    del phresh_data
    gc.collect()
except Exception as e:
    import traceback
    print(f"Error processing PhreshPhish: {e}")
    traceback.print_exc()

# 2. Download and Process ealvaradob
print("\n--- Step 2: Downloading ealvaradob/phishing-dataset ---")
eal_data = []
try:
    from datasets import load_dataset
    ds_eal = load_dataset("ealvaradob/phishing-dataset", "webs", split="train", streaming=True)
    for i, row in enumerate(ds_eal):
        if i >= 100:
            break
        
        text = row.get("text", "")
        label = row.get("label", None)
        url = row.get("url", "")
        
        if not url or not text or label is None:
            continue
            
        eal_data.append({
            "Data": url,
            "html": text,
            "Category": 1 if label == 1 else 0
        })
        
    if len(eal_data) == 0:
        print("Dropped entirely: ealvaradob dataset does not contain both URL and HTML raw data.")
    else:
        df_eal = pd.DataFrame(eal_data)
        eal_path = os.path.join(EXTERNAL_DIR, "ealvaradob.csv")
        df_eal.to_csv(eal_path, index=False)
except Exception as e:
    print(f"Error processing ealvaradob: {e}")

print("\n--- Step 3: Benchmarking Architectures ---")
import sys
sys.path.append(".")
from src.features.structural import StructuralProcessor
from src.models.model_factory import ModelFactory
from src.utils.config_loader import ConfigLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

config = ConfigLoader.load_yaml("config/benchmarks.yaml")

def run_benchmark(dataset_name, data_path):
    print(f"\nBenchmarking on {dataset_name} ({data_path})...")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    processor = StructuralProcessor(config)
    print("Extracting 75 structural invariants...")
    X = processor.fit_transform(df)
    y = df['Category'].values
    
    # Train/Test Split (Train and test on that dataset, as requested)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    results = {}
    for model_name in ["structural_rf", "structural_xgb"]:
        print(f"Training {model_name}...")
        model = ModelFactory.create_model(model_name, config)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[model_name] = {
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F1": f1,
            "AUC": auc
        }
        print(f"  {model_name} -> Acc: {acc:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
    
    return results

print("\n--- Step 3: Benchmarking Architectures ---\n")
benchmark_results = {}

# Run Benchmark for PhreshPhish
if os.path.exists(os.path.join(EXTERNAL_DIR, "phreshphish.parquet")):
    benchmark_results["PhreshPhish"] = run_benchmark("PhreshPhish", os.path.join(EXTERNAL_DIR, "phreshphish.parquet"))

# Save results for documentation
with open(os.path.join(EXTERNAL_DIR, "benchmark_results.json"), "w") as f:
    json.dump(benchmark_results, f, indent=4)

print("\nAll done! Results saved to data/raw/external/benchmark_results.json")

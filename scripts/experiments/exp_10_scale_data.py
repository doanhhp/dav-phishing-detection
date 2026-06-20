import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import requests
import pyarrow.parquet as pq
import joblib
from pathlib import Path
from src.features.structural import StructuralProcessor
from src.utils.config_loader import ConfigLoader

EXTERNAL_DIR = "data/raw/external"
os.makedirs(EXTERNAL_DIR, exist_ok=True)

def fetch_and_parse_parquet(url, label_target, max_records, current_count, phresh_data):
    print(f"Downloading {url}...")
    local_path = os.path.join(EXTERNAL_DIR, "temp_scale.parquet")
    try:
        resp = requests.get(url, stream=True)
        if resp.status_code != 200:
            print(f"Failed to fetch {url}: {resp.status_code}")
            return current_count
            
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Streaming {local_path} in safe memory chunks...")
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=100): 
            df_chunk = batch.to_pandas()
            for _, row in df_chunk.iterrows():
                label = row.get("label")
                
                if label_target == 0 and label != "benign": continue
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
        
    try: os.remove(local_path)
    except: pass
    return current_count

def main():
    print("--- Experiment 10: Scaling PhreshPhish to 40,000 Records ---")
    
    phresh_path = os.path.join(EXTERNAL_DIR, "phreshphish_40k.parquet")
    
    if os.path.exists(phresh_path):
        print(f"Loading existing {phresh_path}...")
        df_phresh = pd.read_parquet(phresh_path)
    else:
        phresh_data = []
        
        # Fetch 20,000 Ham
        ham_count = 0
        for idx in range(30):
            if ham_count >= 20000: break
            ham_url = f"https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/train-{idx:03d}.parquet"
            ham_count = fetch_and_parse_parquet(ham_url, 0, 20000, ham_count, phresh_data)
            print(f"Ham Count: {ham_count}/20000")
            
        # Fetch 20,000 Spam
        spam_count = 0
        for idx in range(55, 20, -1):
            if spam_count >= 20000: break
            spam_url = f"https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/train-{idx:03d}.parquet"
            spam_count = fetch_and_parse_parquet(spam_url, 1, 20000, spam_count, phresh_data)
            print(f"Spam Count: {spam_count}/20000")
            
        df_phresh = pd.DataFrame(phresh_data)
        df_phresh = df_phresh.sample(frac=1, random_state=42).reset_index(drop=True)
        df_phresh.to_parquet(phresh_path, index=False)
        print(f"Successfully saved {len(df_phresh)} records to {phresh_path}")
        
    print("\n--- Extracting Structural Features ---")
    config = ConfigLoader.load_yaml("config/benchmarks.yaml")
    processor = StructuralProcessor(config)
    
    print("Processing 40k HTML documents (This will take 15-20 minutes)...")
    X_proc = processor.fit_transform(df_phresh[['Data', 'html']])
    y = df_phresh['Category'].values
    
    print("\nSaving processed features to disk...")
    proc_dir = Path("data/processed/structural_xgb_40k")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(processor, proc_dir / "processor.joblib")
    joblib.dump(X_proc, proc_dir / "X_phresh_40k.joblib")
    joblib.dump(y, proc_dir / "y_phresh_40k.joblib")
    
    print(f"Done! Saved to {proc_dir}")

if __name__ == "__main__":
    main()

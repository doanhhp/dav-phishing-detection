import pandas as pd
import numpy as np
import joblib
import os
import sys
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
import subprocess

def main():
    print("Loading 2021 Data for Retraining...")
    df = pd.read_parquet('data/raw/external/phreshphish_40k.parquet')
    y = df['Category'].values
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    models = ["structural_rf", "structural_xgb", "structural_stacking", "mid_fusion_xgb"]
    
    for m in models:
        print(f"\n--- Training {m} ---")
        processor = FeatureFactory.get_processor("structural", config["models"][m])
            
        print("Extracting features (with blinded URL features)...")
        X = processor.fit_transform(df[['Data', 'html']])
        
        print(f"Training {m}...")
        model = ModelFactory.create_model(m, config["models"][m])
        model.fit(X, y)
        
        # Save processor
        proc_dir = Path(f"data/processed/{m}")
        proc_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(processor, proc_dir / "processor.joblib")
        
        # Save model
        mod_dir = Path(f"experiments/{m}")
        mod_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(mod_dir / "model"))
        print(f"Saved {m}")

    print("\n--- RE-EVALUATING ON 2026 OOD DATASET ---")
    datasets = {
        "2026_ood": "data/processed/standardized/ood_dataset.parquet"
    }
    
    for m in models:
        print(f"\nEvaluating {m}...")
        cmd = [
            ".venv\\Scripts\\python.exe",
            "scripts/experiments/test_zero_shot.py",
            m,
            "config/benchmarks.yaml",
            "--url_path",
            datasets["2026_ood"],
            "--html_path",
            datasets["2026_ood"]
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        process.wait()
        
    print("\nALL DONE.")

if __name__ == "__main__":
    main()

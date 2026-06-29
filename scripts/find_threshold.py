import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import yaml
import joblib
import pandas as pd
import numpy as np
from src.pipeline import load_data
from src.models.model_factory import ModelFactory
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    df, y = load_data("data/processed/standardized/phreshphish_dataset.parquet")
    X = df[['Data', 'html']]
    
    print("Loading processor...")
    # Load the already extracted features or processor
    processor = joblib.load("data/processed/mid_fusion_xgb/processor.joblib")
    
    # Try to load preprocessed data if it exists for phreshphish, otherwise transform
    try:
        X_processed = joblib.load("data/processed/standardized/phreshphish_processed_structural.joblib")
    except:
        print("Transforming data...")
        X_processed = processor.transform(X)
        joblib.dump(X_processed, "data/processed/standardized/phreshphish_processed_structural.joblib")
        
    print("Loading model...")
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)["models"]["mid_fusion_xgb"]
    model = ModelFactory.create_model("mid_fusion_xgb", config)
    model.load("experiments/mid_fusion_xgb/model")
    
    y_prob = model.predict_proba(X_processed)
    if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
        
    # Test thresholds
    print("Testing thresholds...")
    best_acc = 0
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob > t).astype(int)
        acc = accuracy_score(y, y_pred)
        print(f"Threshold {t:.2f} -> Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_t = t
            
    print(f"\nBest Accuracy: {best_acc:.4f} at Threshold: {best_t:.2f}")

if __name__ == "__main__":
    main()

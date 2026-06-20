import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from src.models.model_factory import ModelFactory
from src.features.factory import FeatureFactory
from src.utils.config_loader import ConfigLoader

import warnings
warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return acc, f1, f2, auc, fpr, fnr

def run_xgboost_benchmark(X, y):
    print("\n--- Running Pruned XGBoost Benchmark ---")
    
    # RFE Logic
    print("Computing RFE selector...")
    full_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    full_model.fit(X, y)
    selector = SelectFromModel(full_model, threshold='median', prefit=True)
    X_pruned = selector.transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_pruned, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Inference Latency
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / len(X_test) * 1000 # ms per sample
    
    y_proba = model.predict_proba(X_test)[:, 1]
    avg_metrics = get_metrics(y_test, y_pred, y_proba)
    
    print(f"XGBoost F2: {avg_metrics[2]:.4f} | FPR: {avg_metrics[4]:.4f} | Latency: {avg_latency:.4f} ms")
    return avg_metrics, avg_latency

def run_cnn_benchmark(X_raw, y):
    print("\n--- Running WebPhish CNN Benchmark ---")
    config = ConfigLoader.load_yaml("config/benchmarks.yaml")
    
    # Need to process features exactly for CNN
    cnn_processor = FeatureFactory.get_processor("multimodal", config)
    print("Extracting features for CNN...")
    X_cnn_proc = cnn_processor.fit_transform(X_raw)
    
    from sklearn.model_selection import train_test_split
    
    # Train WebPhish
    model_config = config.get("models", {}).get("webphish_cnn", {})
    model_config['epochs'] = 10
    model_config['batch_size'] = 128
    
    model = ModelFactory.create_model("webphish_cnn", model_config)
    
    # X_cnn_proc is a list: [url_proc, html_proc]
    # We must split based on indices
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_train_url, X_train_html = X_cnn_proc[0][train_idx], X_cnn_proc[1][train_idx]
    X_test_url, X_test_html = X_cnn_proc[0][test_idx], X_cnn_proc[1][test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print("Training CNN on 80% split...")
    model.fit([X_train_url, X_train_html], y_train)
    
    start_time = time.time()
    y_pred = model.predict([X_test_url, X_test_html])
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / len(test_idx) * 1000 # ms per sample
    
    y_proba = model.predict_proba([X_test_url, X_test_html])[:, 1]
    avg_metrics = get_metrics(y_test, y_pred, y_proba)
    
    print(f"CNN F2: {avg_metrics[2]:.4f} | FPR: {avg_metrics[4]:.4f} | Latency: {avg_latency:.4f} ms")
    return avg_metrics, avg_latency

def main():
    print("Loading scaled 40k PhreshPhish dataset...")
    # Load XGBoost pre-processed data directly for speed
    proc_dir = Path("data/processed/structural_xgb_40k")
    X_xgb = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y = joblib.load(proc_dir / "y_phresh_40k.joblib")
    
    # We also need the raw data for CNN
    print("Loading raw text for CNN processing...")
    df = pd.read_parquet("data/raw/external/phreshphish_40k.parquet")
    X_raw = df[['Data', 'html']]
    
    xgb_metrics, xgb_lat = run_xgboost_benchmark(X_xgb, y)
    cnn_metrics, cnn_lat = run_cnn_benchmark(X_raw, y)
    
    out_dir = Path("docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "advanced_metrics_log.md"
    
    with open(report_path, "w") as f:
        f.write("# Advanced Metrics Comparison\n\n")
        f.write("This report details the operational performance metrics of the Pruned XGBoost model against the WebPhish CNN on the 40k PhreshPhish dataset.\n\n")
        
        f.write("| Metric | Pruned XGBoost | WebPhish CNN |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Accuracy | {xgb_metrics[0]:.4f} | {cnn_metrics[0]:.4f} |\n")
        f.write(f"| F1-Score | {xgb_metrics[1]:.4f} | {cnn_metrics[1]:.4f} |\n")
        f.write(f"| F2-Score | **{xgb_metrics[2]:.4f}** | {cnn_metrics[2]:.4f} |\n")
        f.write(f"| ROC-AUC | {xgb_metrics[3]:.4f} | {cnn_metrics[3]:.4f} |\n")
        f.write(f"| False Positive Rate (FPR) | **{xgb_metrics[4]:.4f}** | {cnn_metrics[4]:.4f} |\n")
        f.write(f"| False Negative Rate (FNR) | {xgb_metrics[5]:.4f} | {cnn_metrics[5]:.4f} |\n")
        f.write(f"| Inference Latency (ms/sample)| **{xgb_lat:.4f} ms** | {cnn_lat:.4f} ms |\n")
        
    print(f"\nReport successfully saved to {report_path}")

if __name__ == "__main__":
    main()

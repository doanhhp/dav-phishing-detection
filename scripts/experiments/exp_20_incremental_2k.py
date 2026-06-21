import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def main():
    print("--- Phase 9: Restricted Incremental Learning (Max 2k) ---")
    
    # 1. Load Pre-processed 2021 Training Data (The 40k/45k historical dataset)
    print("Loading the old 45k historical training data...")
    proc_dir = Path("data/processed/structural_xgb_40k")
    X_train_raw = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y_train = joblib.load(proc_dir / "y_phresh_40k.joblib")
    processor = joblib.load(proc_dir / "processor.joblib")
    
    # Apply RFE Pruning
    print("Applying RFE Pruning...")
    base_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    base_model.fit(X_train_raw, y_train)
    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    X_train_40k = selector.transform(X_train_raw)
    
    # Train Initial Baseline Model on old data
    print("Training frozen Baseline XGBoost on old 45k dataset...")
    model = XGBClassifier(
        n_estimators=250, 
        learning_rate=0.05, 
        max_depth=8, 
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_40k, y_train)
    
    # 2. Load the Zero-Day Streaming Data (OOD 2026)
    print("Loading the new zero-day streaming data...")
    df_url_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url_2026['html'] = df_html_2026.loc[df_url_2026.index, 'Data']
    y_2026 = df_url_2026['Category'].map({'ham': 0, 'spam': 1}).values
    
    X_2026_raw = processor.transform(df_url_2026[['Data', 'html']])
    X_2026 = selector.transform(X_2026_raw)
    
    # 3. Create Stream and Holdout Test Set (20% for testing)
    X_stream, X_test, y_stream, y_test = train_test_split(
        X_2026, y_2026, test_size=0.2, random_state=42, stratify=y_2026
    )
    
    # RESTRICT STREAM TO 2000 SAMPLES MAX
    max_samples = min(2000, len(X_stream))
    X_stream = X_stream[:max_samples]
    y_stream = y_stream[:max_samples]
    
    print(f"Data Split: Stream restricted to MAX {max_samples} samples, Holdout={X_test.shape[0]} samples.")
    
    # Initial Baseline Accuracy on Holdout Test Set
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Initial Zero-Day Holdout Accuracy (0 samples learned): {baseline_acc:.4f}")
    
    # 4. Incremental Learning Loop
    chunk_size = 200
    num_chunks = len(X_stream) // chunk_size
    
    samples_learned = [0]
    accuracies = [baseline_acc]
    
    current_samples = 0
    for i in range(num_chunks + 1):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(X_stream))
        
        if start_idx >= end_idx:
            break
            
        X_chunk = X_stream[start_idx:end_idx]
        y_chunk = y_stream[start_idx:end_idx]
        
        current_estimators = model.get_params()['n_estimators']
        model.set_params(n_estimators=current_estimators + 10)
        
        model.fit(X_chunk, y_chunk, xgb_model=model.get_booster())
        
        current_samples += len(X_chunk)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        
        samples_learned.append(current_samples)
        accuracies.append(acc)
        
        print(f"Learned {current_samples:4d} samples -> Holdout Accuracy: {acc:.4f}")
        
    # 5. Plot the Recovery Curve
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(samples_learned, accuracies, marker='o', linewidth=2, color='#9b59b6', markersize=6)
    plt.axhline(y=baseline_acc, color='#e74c3c', linestyle='--', label='Initial Zero-Day Collapse')
    
    plt.title('Restricted Domain Shift Adaptation (Max 2k Samples)', fontsize=14, pad=15)
    plt.xlabel('Number of Zero-Day Samples Learned', fontsize=12)
    plt.ylabel('Accuracy on Holdout Set', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend()
    
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "incremental_learning_2k.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nExperiment Complete. Plot saved to docs/assets/incremental_learning_2k.png")

if __name__ == "__main__":
    main()

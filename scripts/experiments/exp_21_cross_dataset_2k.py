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
    print("--- Phase 9: Restricted Cross-Dataset Incremental Learning (Max 2k) ---")
    
    # 1. Load OLD 2021 Dataset (URL.xlsx 11k)
    print("Loading RAW OLD 2021 Data (11k samples)...")
    from src.features.structural import StructuralProcessor
    from src.utils.config_loader import ConfigLoader
    
    df_url_old = pd.read_excel("data/raw/URL.xlsx")
    df_html_old = pd.read_excel("data/raw/html.xlsx")
    df_url_old['html'] = df_html_old['Data']
    df_old = df_url_old[['Data', 'html']].copy()
    y_old = df_url_old['Category'].map({'ham': 0, 'spam': 1}).values
    
    print("Fitting modern StructuralProcessor on OLD 11k Data...")
    config = ConfigLoader.load_yaml("config/benchmarks.yaml")
    processor = StructuralProcessor(config)
    X_old_raw = processor.fit_transform(df_old)
    
    # Apply RFE Pruning
    print("Applying RFE Pruning to OLD dataset...")
    base_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    base_model.fit(X_old_raw, y_old)
    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    X_old_pruned = selector.transform(X_old_raw)
    
    # Train Initial Baseline Model on OLD Data
    print("Training frozen Baseline XGBoost on OLD dataset...")
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
    model.fit(X_old_pruned, y_old)
    
    # 2. Load NEWER PhreshPhish 40k Data
    print("\nLoading NEWER PhreshPhish 40k Data...")
    df_phresh = pd.read_parquet("data/raw/external/phreshphish_40k.parquet")
    y_phresh = df_phresh['Category'].values
    
    # Transform using the OLD processor fitted on the 11k dataset
    X_phresh_raw = processor.transform(df_phresh[['Data', 'html']])
    X_phresh = selector.transform(X_phresh_raw)
    
    # 3. Create Stream and Holdout Test Set (20% for testing ~ 8k samples)
    X_stream, X_test, y_stream, y_test = train_test_split(
        X_phresh, y_phresh, test_size=0.2, random_state=42, stratify=y_phresh
    )
    
    # RESTRICT STREAM TO 2000 SAMPLES MAX
    max_samples = min(2000, len(X_stream))
    X_stream = X_stream[:max_samples]
    y_stream = y_stream[:max_samples]
    
    print(f"\nPhreshPhish Split: Stream restricted to MAX {max_samples} samples, Holdout={X_test.shape[0]} samples.")
    
    # Initial Baseline Accuracy on PhreshPhish Holdout Test Set
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Initial Cross-Dataset Holdout Accuracy (0 samples learned): {baseline_acc:.4f}")
    
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
        
        print(f"Learned {current_samples:5d} samples -> Holdout Accuracy: {acc:.4f}")
        
    # 5. Plot the Recovery Curve
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(samples_learned, accuracies, marker='o', linewidth=2, color='#e67e22', markersize=6)
    plt.axhline(y=baseline_acc, color='#e74c3c', linestyle='--', label='Initial Cross-Dataset Collapse')
    
    plt.title('Restricted Cross-Dataset Adaptation (Max 2k Samples)', fontsize=14, pad=15)
    plt.xlabel('Number of PhreshPhish Samples Learned', fontsize=12)
    plt.ylabel('Accuracy on PhreshPhish Holdout Set', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend()
    
    out_dir = Path("docs/assets")
    plt.savefig(out_dir / "cross_dataset_incremental_2k.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nExperiment Complete. Plot saved to docs/assets/cross_dataset_incremental_2k.png")

if __name__ == "__main__":
    main()

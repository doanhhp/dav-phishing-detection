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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from src.features.structural import StructuralProcessor
from src.utils.config_loader import ConfigLoader
import warnings

warnings.filterwarnings('ignore')

def main():
    print("--- Phase 10: Random Forest Few-Shot Data Mixing Retraining ---")
    
    # 1. Load OLD 2021 Dataset (URL.xlsx 11k)
    print("Loading RAW OLD 2021 Data (11k samples)...")
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
    
    # 2. Load NEWER PhreshPhish 40k Data
    print("\nLoading NEWER PhreshPhish 40k Data...")
    df_phresh = pd.read_parquet("data/raw/external/phreshphish_40k.parquet")
    y_phresh = df_phresh['Category'].values
    
    X_phresh_raw = processor.transform(df_phresh[['Data', 'html']])
    X_phresh = selector.transform(X_phresh_raw)
    
    # 3. Create Pool and Holdout Test Set (20% for testing ~ 8k samples)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_phresh, y_phresh, test_size=0.2, random_state=42, stratify=y_phresh
    )
    
    print(f"\nPhreshPhish Split: Pool for Injection={X_pool.shape[0]} samples, Holdout={X_test.shape[0]} samples.")
    
    # 4. Few-Shot Data Mixing Retraining Loop
    injection_sizes = [0, 50, 100, 200, 500, 1000, 2000, 5000]
    accuracies = []
    
    for size in injection_sizes:
        print(f"\n--- Injecting {size} PhreshPhish samples into the old 11k dataset ---")
        
        if size == 0:
            X_mixed = X_old_pruned
            y_mixed = y_old
        else:
            # Sample 'size' records from the pool
            # We use a fixed random state to ensure consistent subsets
            X_inject, _, y_inject, _ = train_test_split(
                X_pool, y_pool, train_size=size, random_state=42, stratify=y_pool
            )
            
            # Mix old data with the injected data
            X_mixed = np.vstack((X_old_pruned, X_inject))
            y_mixed = np.concatenate((y_old, y_inject))
            
        print(f"Total Retraining Data Size: {X_mixed.shape[0]}")
        
        # Fully Retrain Random Forest
        rf = RandomForestClassifier(
            n_estimators=250, 
            max_depth=15, 
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_mixed, y_mixed)
        
        # Test on Holdout
        acc = accuracy_score(y_test, rf.predict(X_test))
        accuracies.append(acc)
        
        print(f"Retrained RF Accuracy on Holdout: {acc * 100:.2f}%")
        
    # 5. Plot the Few-Shot Curve
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(injection_sizes, accuracies, marker='s', linewidth=2, color='#8e44ad', markersize=6)
    plt.axhline(y=accuracies[0], color='#e74c3c', linestyle='--', label=f'Baseline (0 Injections) = {accuracies[0]*100:.2f}%')
    
    plt.title('Random Forest: Few-Shot Data Mixing Retraining', fontsize=14, pad=15)
    plt.xlabel('Number of Target Domain Samples Injected', fontsize=12)
    plt.ylabel('Accuracy on Target Domain Holdout Set', fontsize=12)
    plt.ylim(0.7, 1.0)
    
    # Annotate points
    for i, txt in enumerate(accuracies):
        plt.annotate(f"{txt*100:.1f}%", (injection_sizes[i], accuracies[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.legend()
    
    out_dir = Path("docs/assets")
    plt.savefig(out_dir / "rf_few_shot_retrain.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nExperiment Complete. Plot saved to docs/assets/rf_few_shot_retrain.png")

if __name__ == "__main__":
    main()

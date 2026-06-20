import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from src.features.structural import StructuralProcessor
from src.utils.config_loader import ConfigLoader

# Suppress PyArrow warning
import warnings
warnings.filterwarnings('ignore')

def run_kfold_evaluation(dataset_name, X, y, n_splits=5):
    print(f"\nEvaluating {dataset_name} ({len(X)} samples) with {n_splits}-Fold CV...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, recall_scores, auc_scores = [], [], []
    
    config = ConfigLoader.load_yaml("config/benchmarks.yaml")
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Initialize processor
        processor = StructuralProcessor(config)
        X_train_proc = processor.fit_transform(X_train)
        X_test_proc = processor.transform(X_test)
        
        # Train XGBoost
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
        model.fit(X_train_proc, y_train)
        
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        
        acc_scores.append(accuracy_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
        
        print(f"  Fold {fold}: Acc {acc_scores[-1]:.4f} | Recall {recall_scores[-1]:.4f} | AUC {auc_scores[-1]:.4f}")
        fold += 1
        
    print(f"--> Average {dataset_name}: Acc: {np.mean(acc_scores):.4f}, Recall: {np.mean(recall_scores):.4f}, AUC: {np.mean(auc_scores):.4f}")

def main():
    print("--- Experiment 8: XPath Sequences & Cloaking Indicators ---")
    
    # 1. Dataset 1: Historical 2021 (Too large for fast CV, using 5000 sample)
    print("\nLoading Dataset 1: Historical 2021 (Subsampled to 5000 for speed)...")
    df_url_2021 = pd.read_excel("data/raw/URL.xlsx").sample(5000, random_state=42)
    df_html_2021 = pd.read_excel("data/raw/html.xlsx")
    df_url_2021['html'] = df_html_2021.loc[df_url_2021.index, 'Data']
    y_2021 = df_url_2021['Category'].map({'ham': 0, 'spam': 1}).values
    X_2021 = df_url_2021[['Data', 'html']]
    run_kfold_evaluation("Historical 2021", X_2021, y_2021)
    del df_url_2021, df_html_2021
    
    # 2. Dataset 2: Zero-Day 2026 OOD (Full dataset)
    print("\nLoading Dataset 2: Zero-Day 2026 OOD...")
    df_url_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url_2026['html'] = df_html_2026.loc[df_url_2026.index, 'Data']
    y_2026 = df_url_2026['Category'].map({'ham': 0, 'spam': 1}).values
    X_2026 = df_url_2026[['Data', 'html']]
    run_kfold_evaluation("Zero-Day 2026", X_2026, y_2026)
    del df_url_2026, df_html_2026
    
    # 3. Dataset 3: PhreshPhish (External)
    print("\nLoading Dataset 3: PhreshPhish External Benchmark...")
    df_phresh = pd.read_parquet("data/raw/external/phreshphish.parquet")
    y_phresh = df_phresh['Category'].values
    X_phresh = df_phresh[['Data', 'html']]
    run_kfold_evaluation("PhreshPhish", X_phresh, y_phresh)

if __name__ == "__main__":
    main()

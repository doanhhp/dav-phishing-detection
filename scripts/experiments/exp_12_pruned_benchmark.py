import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from src.features.structural import StructuralProcessor
from src.utils.config_loader import ConfigLoader

import warnings
warnings.filterwarnings('ignore')

def run_pruned_kfold(dataset_name, X_raw, y_target, processor, selector):
    print(f"\nEvaluating {dataset_name} ({len(X_raw)} samples) with 5-Fold CV...")
    
    # Process features
    X_proc = processor.transform(X_raw)
    
    # Prune features
    X_pruned = selector.transform(X_proc)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores, recall_scores, auc_scores = [], [], []
    
    fold = 1
    for train_idx, test_idx in skf.split(X_pruned, y_target):
        X_train, X_test = X_pruned[train_idx], X_pruned[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]
        
        # Using the tuned parameters from Phase 4
        model = XGBClassifier(
            n_estimators=250, 
            learning_rate=0.05, 
            max_depth=8, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc_scores.append(accuracy_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
        fold += 1
        
    print(f"--> Average {dataset_name}: Acc: {np.mean(acc_scores):.4f}, Recall: {np.mean(recall_scores):.4f}, AUC: {np.mean(auc_scores):.4f}")


def main():
    print("--- Experiment 12: Pruned Feature Set on All Datasets ---")
    
    # 1. Establish the Pruning Selector based on 40k PhreshPhish
    print("\nLoading massive 40k dataset to recreate RFE pruning selector...")
    proc_dir = Path("data/processed/structural_xgb_40k")
    X_phresh = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y_phresh = joblib.load(proc_dir / "y_phresh_40k.joblib")
    processor = joblib.load(proc_dir / "processor.joblib")
    
    full_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    full_model.fit(X_phresh, y_phresh)
    
    selector = SelectFromModel(full_model, threshold='median', prefit=True)
    kept_features = selector.get_support(indices=True).shape[0]
    print(f"Selector ready. Reducing from {X_phresh.shape[1]} to {kept_features} features.")
    
    del X_phresh, y_phresh, full_model
    
    # 2. Historical 2021 Dataset
    print("\nLoading Dataset 1: Historical 2021...")
    df_url_2021 = pd.read_excel("data/raw/URL.xlsx").sample(5000, random_state=42)
    df_html_2021 = pd.read_excel("data/raw/html.xlsx")
    df_url_2021['html'] = df_html_2021.loc[df_url_2021.index, 'Data']
    y_2021 = df_url_2021['Category'].map({'ham': 0, 'spam': 1}).values
    X_2021 = df_url_2021[['Data', 'html']]
    run_pruned_kfold("Historical 2021", X_2021, y_2021, processor, selector)
    del df_url_2021, df_html_2021
    
    # 3. Zero-Day 2026 Dataset
    print("\nLoading Dataset 2: Zero-Day 2026 OOD...")
    df_url_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url_2026['html'] = df_html_2026.loc[df_url_2026.index, 'Data']
    y_2026 = df_url_2026['Category'].map({'ham': 0, 'spam': 1}).values
    X_2026 = df_url_2026[['Data', 'html']]
    run_pruned_kfold("Zero-Day 2026", X_2026, y_2026, processor, selector)
    del df_url_2026, df_html_2026
    
    # 4. PhreshPhish is already computed in Exp 11, but we can re-state it.
    print("\nNote: PhreshPhish 40k Performance previously evaluated as:")
    print("--> Average PhreshPhish: Acc: 0.9637, Recall: 0.9590, AUC: 0.9941")

if __name__ == "__main__":
    main()

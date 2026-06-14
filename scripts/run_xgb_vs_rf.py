import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from src.models.structural_rf import Structural_RF
from src.models.structural_xgb import Structural_XGB

def main():
    print("--- 1. Loading Original Training Data (Historical) ---")
    df_train_url = pd.read_excel('data/raw/URL.xlsx')
    df_train_html = pd.read_excel('data/raw/html.xlsx')
    
    if 'Label' in df_train_url.columns:
        df_train_url = df_train_url.rename(columns={'Label': 'Category'})
    
    y_train = df_train_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    print("Extracting Structural Features for Training...")
    proc_path = "data/processed/structural_rf/processor.joblib"
    if os.path.exists(proc_path):
        struct_proc = joblib.load(proc_path)
        df_train_raw = df_train_url[['Data']].copy()
        df_train_raw['html'] = df_train_html['Data']
        X_train = struct_proc.transform(df_train_raw)
    else:
        from src.features.structural import StructuralProcessor
        struct_proc = StructuralProcessor({})
        df_train_raw = df_train_url[['Data']].copy()
        df_train_raw['html'] = df_train_html['Data']
        X_train = struct_proc.fit_transform(df_train_raw)

    print("\n--- 2. Building & Training Models ---")
    
    # Train Random Forest
    print(f"Training Structural RF on all {X_train.shape[0]} historical samples...")
    rf = Structural_RF(config={'n_estimators': 100})
    rf.fit(X_train, y_train)

    # Train XGBoost
    print(f"Training Structural XGBoost on all {X_train.shape[0]} historical samples...")
    xgb_model = Structural_XGB(config={'n_estimators': 200, 'learning_rate': 0.1})
    xgb_model.fit(X_train, y_train)

    print("\n--- 3. Evaluating on Zero-Shot OOD Dataset (1.8k New Samples) ---")
    df_ood_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_ood_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    print("Extracting OOD Structural Features...")
    df_ood_raw = df_ood_url[['Data']].copy()
    df_ood_raw['html'] = df_ood_html['Data']
    X_ood = struct_proc.transform(df_ood_raw)

    print("\n--- 4. Calculating Predictions ---")
    # Random Forest Predictions
    rf_preds = rf.predict(X_ood)
    rf_probs = rf.predict_proba(X_ood)[:, 1]
    
    # XGBoost Predictions
    xgb_preds = xgb_model.predict(X_ood)
    xgb_probs = xgb_model.predict_proba(X_ood)[:, 1]

    print("\n--- 5. Results & Comparisons ---")
    rf_acc = accuracy_score(y_ood, rf_preds)
    rf_auc = roc_auc_score(y_ood, rf_probs)
    rf_f1 = f1_score(y_ood, rf_preds)
    
    xgb_acc = accuracy_score(y_ood, xgb_preds)
    xgb_auc = roc_auc_score(y_ood, xgb_probs)
    xgb_f1 = f1_score(y_ood, xgb_preds)
    
    print(f"Random Forest -> Acc: {rf_acc:.4f} | AUC: {rf_auc:.4f} | F1: {rf_f1:.4f}")
    print(f"XGBoost       -> Acc: {xgb_acc:.4f} | AUC: {xgb_auc:.4f} | F1: {xgb_f1:.4f}")

    print("\nDetailed Reports:")
    print("Random Forest:")
    print(classification_report(y_ood, rf_preds))
    print("XGBoost:")
    print(classification_report(y_ood, xgb_preds))

if __name__ == "__main__":
    main()

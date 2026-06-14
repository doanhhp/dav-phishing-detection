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
from sklearn.model_selection import train_test_split

def main():
    print("--- 1. Loading Mix of Old and New Data (20/80 Split) ---")
    
    # Load 200 Old Records
    df_old_url = pd.read_excel('data/raw/URL.xlsx').head(200)
    df_old_html = pd.read_excel('data/raw/html.xlsx').head(200)
    if 'Label' in df_old_url.columns:
        df_old_url = df_old_url.rename(columns={'Label': 'Category'})
    
    # Load All New Records
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    # Take 800 New Records for training, leave the rest for testing
    df_new_train_url = df_new_url.iloc[:800].copy()
    df_new_train_html = df_new_html.iloc[:800].copy()
    
    df_new_test_url = df_new_url.iloc[800:].copy()
    df_new_test_html = df_new_html.iloc[800:].copy()
    
    # Combine training sets (200 old + 800 new = 1000 total)
    df_train_url = pd.concat([df_old_url, df_new_train_url], ignore_index=True)
    df_train_html = pd.concat([df_old_html, df_new_train_html], ignore_index=True)
    
    y_train = df_train_url['Category'].map({'ham': 0, 'spam': 1}).values
    y_test = df_new_test_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    print(f"Training Size: {len(y_train)} (20% Old, 80% New)")
    print(f"Testing Size: {len(y_test)} (100% Unseen New Data)")

    print("\n--- 2. Extracting Structural Features ---")
    proc_path = "data/processed/structural_rf/processor.joblib"
    if os.path.exists(proc_path):
        struct_proc = joblib.load(proc_path)
    else:
        from src.features.structural import StructuralProcessor
        struct_proc = StructuralProcessor({})
        
    df_train_raw = df_train_url[['Data']].copy()
    df_train_raw['html'] = df_train_html['Data']
    X_train = struct_proc.transform(df_train_raw)
    
    df_test_raw = df_new_test_url[['Data']].copy()
    df_test_raw['html'] = df_new_test_html['Data']
    X_test = struct_proc.transform(df_test_raw)

    print("\n--- 3. Building & Training Models ---")
    
    print("Training Structural RF on the 20/80 mix...")
    rf = Structural_RF(config={'n_estimators': 100})
    rf.fit(X_train, y_train)

    print("Training Structural XGBoost on the 20/80 mix...")
    xgb_model = Structural_XGB(config={'n_estimators': 100, 'learning_rate': 0.1})
    xgb_model.fit(X_train, y_train)

    print("\n--- 4. Evaluating on Remaining Zero-Shot Data ---")
    # Random Forest Predictions
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    # XGBoost Predictions
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    print("\n--- 5. Results & Comparisons ---")
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_f1 = f1_score(y_test, rf_preds)
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    print(f"Random Forest (20/80 Mix) -> Acc: {rf_acc:.4f} | AUC: {rf_auc:.4f} | F1: {rf_f1:.4f}")
    print(f"XGBoost       (20/80 Mix) -> Acc: {xgb_acc:.4f} | AUC: {xgb_auc:.4f} | F1: {xgb_f1:.4f}")

    print("\nDetailed Reports:")
    print("Random Forest:")
    print(classification_report(y_test, rf_preds))
    print("XGBoost:")
    print(classification_report(y_test, xgb_preds))

if __name__ == "__main__":
    main()

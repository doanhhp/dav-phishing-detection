import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from src.models.structural_rf import Structural_RF
from src.models.structural_xgb import Structural_XGB
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Original Training Data (Historical) ---")
    df_train_url = pd.read_excel('data/raw/URL.xlsx')
    df_train_html = pd.read_excel('data/raw/html.xlsx')
    
    if 'Label' in df_train_url.columns:
        df_train_url = df_train_url.rename(columns={'Label': 'Category'})
    
    y = df_train_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    df_raw = df_train_url[['Data']].copy()
    df_raw['html'] = df_train_html['Data']
    
    print("--- 2. Performing 80/20 Train/Test Split ---")
    df_train, df_test, y_train, y_test = train_test_split(
        df_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 3. Extracting Structural Features (No Leakage) ---")
    processor = StructuralProcessor({})
    # Fit on train, transform on test
    X_train = processor.fit_transform(df_train)
    X_test = processor.transform(df_test)

    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    print("\n--- 4. Building & Training Models ---")
    
    # Train Random Forest
    print("Training Structural RF...")
    rf = Structural_RF(config={'n_estimators': 100})
    rf.fit(X_train, y_train)

    # Train XGBoost
    print("Training Structural XGBoost (Hyper-Optimized)...")
    xgb_model = Structural_XGB(config={'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 12})
    xgb_model.fit(X_train, y_train)

    print("\n--- 5. Evaluating on Historical Test Set ---")
    
    # Random Forest Predictions
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    # XGBoost Predictions
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_f1 = f1_score(y_test, rf_preds)
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    print(f"Random Forest (In-Domain 2021) -> Acc: {rf_acc:.4f} | AUC: {rf_auc:.4f} | F1: {rf_f1:.4f}")
    print(f"XGBoost (In-Domain 2021)       -> Acc: {xgb_acc:.4f} | AUC: {xgb_auc:.4f} | F1: {xgb_f1:.4f}")

    print("\nDetailed Reports:")
    print("Random Forest:")
    print(classification_report(y_test, rf_preds))
    print("XGBoost:")
    print(classification_report(y_test, xgb_preds))

if __name__ == "__main__":
    main()

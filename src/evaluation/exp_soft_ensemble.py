"""Experiment: Soft Ensemble (Stacking) Meta-Classifier"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import sys

def main():
    print("--- 1. Loading Base Models and Processors ---")
    url_model_path = r"D:\Desktop\PhishingDetection\experiments\url_rf\model.joblib"
    url_proc_path = r"D:\Desktop\PhishingDetection\data\processed\url_rf\processor.joblib"
    struct_model_path = r"D:\Desktop\PhishingDetection\experiments\structural_rf\model.joblib"
    struct_proc_path = r"D:\Desktop\PhishingDetection\data\processed\structural_rf\processor.joblib"

    url_rf = joblib.load(url_model_path)
    url_proc = joblib.load(url_proc_path)
    struct_rf = joblib.load(struct_model_path)
    struct_proc = joblib.load(struct_proc_path)

    print("--- 2. Loading Test Split (for Meta-Classifier Training) ---")
    url_X_test = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\url_rf\X_test.joblib")
    url_y_test = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\url_rf\y_test.joblib")
    
    struct_X_test = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\structural_rf\X_test.joblib")
    struct_y_test = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\structural_rf\y_test.joblib")

    # Verify alignment
    if not np.array_equal(url_y_test, struct_y_test):
        print("WARNING: Test splits are not aligned! The meta-classifier might not learn correctly.")
        
    print("Generating base model probabilities...")
    url_probs = url_rf.predict_proba(url_X_test)[:, 1]
    struct_probs = struct_rf.predict_proba(struct_X_test)[:, 1]

    # Combine into Meta-Features (Shape: [n_samples, 2])
    X_meta_train = np.column_stack((url_probs, struct_probs))
    y_meta_train = url_y_test

    print("--- 3. Training Meta-Classifier (Logistic Regression) ---")
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta_train, y_meta_train)
    
    print(f"Meta-Classifier Coefficients: URL Weight = {meta_clf.coef_[0][0]:.4f}, Structural Weight = {meta_clf.coef_[0][1]:.4f}")

    print("--- 4. Evaluating on Zero-Shot OOD Dataset ---")
    ood_url_path = r"D:\Desktop\PhishingDetection\data\raw\OOD_URL.xlsx"
    ood_html_path = r"D:\Desktop\PhishingDetection\data\raw\OOD_html.xlsx"
    
    df_ood_url = pd.read_excel(ood_url_path)
    df_ood_html = pd.read_excel(ood_html_path)
    
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    print("Extracting OOD Features...")
    # URL Processor expects a DataFrame with 'Data' column
    url_X_ood_raw = df_ood_url[['Data']]
    url_X_ood = url_proc.transform(url_X_ood_raw)
    
    # Structural Processor expects 'Data' (URL) and 'html'
    df_struct_raw = df_ood_url[['Data']].copy()
    df_struct_raw['html'] = df_ood_html['Data']
    struct_X_ood = struct_proc.transform(df_struct_raw)

    print("Generating OOD Base Probabilities...")
    ood_url_probs = url_rf.predict_proba(url_X_ood)[:, 1]
    ood_struct_probs = struct_rf.predict_proba(struct_X_ood)[:, 1]
    
    X_meta_ood = np.column_stack((ood_url_probs, ood_struct_probs))

    print("Predicting with Meta-Classifier...")
    ood_preds = meta_clf.predict(X_meta_ood)
    
    acc = accuracy_score(y_ood, ood_preds)
    print(f"\nFINAL SOFT ENSEMBLE ZERO-SHOT OOD ACCURACY: {acc:.4f}")
    print(classification_report(y_ood, ood_preds))

if __name__ == "__main__":
    main()

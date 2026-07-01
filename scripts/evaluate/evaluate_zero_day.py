import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import warnings

warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "F2-Score": fbeta_score(y_true, y_pred, beta=2),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }

def main():
    print("--- FINAL EVALUATION: True Zero-Day Domain Shift Test ---")
    print("Training Domain: 2021 (PhreshPhish 40k)")
    print("Testing Domain : 2026 (Newly Crawled Zero-Day Dataset)\n")
    
    # 1. Load Pre-processed 2021 Training Data
    proc_dir = Path("data/processed/structural_xgb_40k")
    X_train_raw = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y_train = joblib.load(proc_dir / "y_phresh_40k.joblib")
    processor = joblib.load(proc_dir / "processor.joblib")
    
    # Apply RFE Pruning
    print("Applying RFE Pruning Mask (Reducing 103 -> 52 features)...")
    base_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    base_model.fit(X_train_raw, y_train)
    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    X_train = selector.transform(X_train_raw)
    
    # 2. Load and Process 2026 Testing Data
    print("Loading 2026 Zero-Day Testing Data...")
    df_url_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url_2026['html'] = df_html_2026.loc[df_url_2026.index, 'Data']
    df_2026 = df_url_2026[['Data', 'html']].copy()
    y_2026 = df_url_2026['Category'].map({'ham': 0, 'spam': 1}).values
    
    X_2026_raw = processor.transform(df_2026)
    X_2026 = selector.transform(X_2026_raw)
    
    # 3. Train the Final Baseline Pruned XGBoost Model
    # (The highly resilient parameters we discovered in Phase 4)
    print("\nTraining Final Baseline XGBoost Model on 2021 Data...")
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
    model.fit(X_train, y_train)
    
    # 4. Evaluate on 2026 Zero-Day Data
    print("\nExecuting True Zero-Day Evaluation on 2026 Dataset...")
    y_pred = model.predict(X_2026)
    y_proba = model.predict_proba(X_2026)[:, 1]
    
    metrics = get_metrics(y_2026, y_pred, y_proba)
    
    print("\n--- FINAL ZERO-DAY RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_2026, y_pred))

if __name__ == "__main__":
    main()

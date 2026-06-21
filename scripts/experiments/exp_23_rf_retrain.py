import sys
import os
import time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import warnings

warnings.filterwarnings('ignore')

def main():
    print("--- Phase 10: Retraining Speed & Accuracy (RF vs XGBoost) ---")
    
    # 1. Load Pre-processed PhreshPhish Data (40k)
    print("Loading Pre-processed PhreshPhish Data (40k)...")
    proc_dir = Path("data/processed/structural_xgb_40k")
    X_raw = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y = joblib.load(proc_dir / "y_phresh_40k.joblib")
    
    # Apply RFE Pruning to match exactly what we did before
    print("Applying RFE Pruning...")
    base_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    base_model.fit(X_raw, y)
    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    X_pruned = selector.transform(X_raw)
    
    # 2. Train/Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pruned, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # 3. Full Retrain: XGBoost
    print("\n--- XGBoost (Full Retrain) ---")
    xgb = XGBClassifier(
        n_estimators=250, 
        learning_rate=0.05, 
        max_depth=8, 
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    
    start_time = time.time()
    xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - start_time
    
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    
    print(f"Training Time : {xgb_train_time:.4f} seconds")
    print(f"Final Accuracy: {xgb_acc * 100:.2f}%")
    
    # 4. Full Retrain: Random Forest
    print("\n--- Random Forest (Full Retrain) ---")
    rf = RandomForestClassifier(
        n_estimators=250, 
        max_depth=15, 
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Training Time : {rf_train_time:.4f} seconds")
    print(f"Final Accuracy: {rf_acc * 100:.2f}%")
    
    # 5. Conclusion
    print("\n--- Conclusion ---")
    if rf_train_time < xgb_train_time:
        speedup = xgb_train_time / rf_train_time
        print(f"Random Forest was {speedup:.1f}x FASTER to retrain.")
    else:
        speedup = rf_train_time / xgb_train_time
        print(f"XGBoost was {speedup:.1f}x FASTER to retrain.")
        
    diff = abs(xgb_acc - rf_acc) * 100
    if rf_acc > xgb_acc:
        print(f"Random Forest was MORE accurate by {diff:.2f}%.")
    else:
        print(f"XGBoost was MORE accurate by {diff:.2f}%.")

if __name__ == "__main__":
    main()

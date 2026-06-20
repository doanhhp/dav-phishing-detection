import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

def main():
    print("--- Experiment 11: RFE & Hyperparameter Tuning on 40k Data ---")
    
    proc_dir = Path("data/processed/structural_xgb_40k")
    
    print("Loading pre-processed features (X_phresh_40k.joblib) from disk...")
    try:
        X = joblib.load(proc_dir / "X_phresh_40k.joblib")
        y = joblib.load(proc_dir / "y_phresh_40k.joblib")
        processor = joblib.load(proc_dir / "processor.joblib")
        feature_names = processor.get_feature_names()
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return
        
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Phase 1: Baseline Evaluation (All Features) ---")
    base_acc, base_rec, base_auc = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, 
            random_state=42, eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        base_acc.append(accuracy_score(y_test, y_pred))
        base_rec.append(recall_score(y_test, y_pred))
        base_auc.append(roc_auc_score(y_test, y_proba))
        
    print(f"Baseline -> Acc: {np.mean(base_acc):.4f} | Recall: {np.mean(base_rec):.4f} | AUC: {np.mean(base_auc):.4f}")
    
    print("\n--- Phase 2: Feature Elimination (Noise Reduction) ---")
    # Train full model to get importances
    full_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    full_model.fit(X, y)
    
    importances = full_model.feature_importances_
    # Calculate threshold (e.g., median importance or a custom threshold to drop bottom noise)
    # We will use SelectFromModel with '0.75*median' or 'median' to prune
    selector = SelectFromModel(full_model, threshold='median', prefit=True)
    
    X_pruned = selector.transform(X)
    kept_features_idx = selector.get_support(indices=True)
    dropped_features = [feature_names[i] for i in range(len(feature_names)) if i not in kept_features_idx]
    
    print(f"Original Features: {X.shape[1]}")
    print(f"Features Kept: {X_pruned.shape[1]}")
    print(f"Dropped {len(dropped_features)} features (Noise).")
    
    print("\n--- Phase 3: Pruned Evaluation ---")
    pruned_acc, pruned_rec, pruned_auc = [], [], []
    for train_idx, test_idx in skf.split(X_pruned, y):
        X_train, X_test = X_pruned[train_idx], X_pruned[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Slightly tune hyperparameters for massive dataset
        model = XGBClassifier(
            n_estimators=250,        # Increased trees for 40k samples
            learning_rate=0.05,      # Lower learning rate for better generalization
            max_depth=8,             # Deeper trees
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        pruned_acc.append(accuracy_score(y_test, y_pred))
        pruned_rec.append(recall_score(y_test, y_pred))
        pruned_auc.append(roc_auc_score(y_test, y_proba))
        
    print(f"Pruned + Tuned -> Acc: {np.mean(pruned_acc):.4f} | Recall: {np.mean(pruned_rec):.4f} | AUC: {np.mean(pruned_auc):.4f}")

if __name__ == "__main__":
    main()

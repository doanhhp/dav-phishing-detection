import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.features.structural import StructuralProcessor
from src.models.structural_xgb import Structural_XGB

def main():
    print("--- 1. Loading New Zero-Day Dataset (OOD) ---")
    df_ood_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_ood_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    if 'Label' in df_ood_url.columns:
        df_ood_url = df_ood_url.rename(columns={'Label': 'Category'})
    
    y = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    df_raw = df_ood_url[['Data']].copy()
    df_raw['html'] = df_ood_html['Data']
    
    print("--- 2. Extracting Structural Features ---")
    processor = StructuralProcessor({})
    X = processor.fit_transform(df_raw)
    
    print("\n--- 3. Performing 5-Fold Cross Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores = []
    auc_scores = []
    f1_scores = []
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        xgb_model = Structural_XGB(config={'n_estimators': 200, 'learning_rate': 0.1})
        xgb_model.fit(X_train, y_train)
        
        preds = xgb_model.predict(X_test)
        probs = xgb_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        print(f"Fold {fold} - Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
        
        acc_scores.append(acc)
        auc_scores.append(auc)
        f1_scores.append(f1)
        fold += 1
        
    print("\n--- 4. Final 5-Fold CV Results ---")
    print(f"Mean Accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"Mean AUC:      {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"Mean F1:       {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

if __name__ == "__main__":
    main()

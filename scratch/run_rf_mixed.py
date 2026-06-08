import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def main():
    print("--- 1. Loading Historical Data ---")
    orig_url = pd.read_excel(project_root / "data/raw/URL.xlsx")
    orig_html = pd.read_excel(project_root / "data/raw/html.xlsx")
    orig_url['html'] = orig_html.loc[orig_url.index, 'Data']
    
    # Subsample 3000 historical samples
    df_hist = orig_url.sample(3000, random_state=42)
    y_hist = df_hist['Category'].map({'ham': 0, 'spam': 1}).values
    
    print("--- 2. Loading OOD Data ---")
    df_ood_url = pd.read_excel(project_root / "data/raw/OOD_URL.xlsx")
    df_ood_html = pd.read_excel(project_root / "data/raw/OOD_html.xlsx")
    df_ood_url['html'] = df_ood_html['Data']
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    print("--- 3. Extracting Features ---")
    processor = joblib.load(project_root / "data/processed/structural_rf/processor.joblib")
    
    X_hist_proc = processor.transform(df_hist[['Data', 'html']])
    X_ood_proc = processor.transform(df_ood_url[['Data', 'html']])
    
    sample_sizes = [10, 20, 50, 100, 200]
    n_seeds = 3
    results = []
    
    print("--- 4. Running Mixed Retraining for RF ---")
    for n_samples in sample_sizes:
        acc_list = []
        for seed in range(n_seeds):
            indices = np.arange(len(y_ood))
            idx_train_new, idx_test_new, y_train_new, y_test_new = train_test_split(
                indices, y_ood, train_size=n_samples, random_state=seed, stratify=y_ood
            )
            
            X_train_new = X_ood_proc[idx_train_new]
            X_test_new = X_ood_proc[idx_test_new]
            
            # Mix
            X_train_mixed = np.vstack([X_hist_proc, X_train_new])
            y_train_mixed = np.concatenate([y_hist, y_train_new])
            
            m_rf = RandomForestClassifier(n_estimators=50, random_state=seed)
            m_rf.fit(X_train_mixed, y_train_mixed)
            
            preds = m_rf.predict(X_test_new)
            acc_list.append(accuracy_score(y_test_new, preds))
            
        mean_acc = np.mean(acc_list)
        print(f"RF Mixed Retrain (3000 + {n_samples}): {mean_acc:.4f}")

if __name__ == "__main__":
    main()

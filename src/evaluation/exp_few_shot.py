"""Experiment: Few-Shot Adaptation Comparison."""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

from pathlib import Path
import sys

# Support running directly from any directory
project_root_sys = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root_sys))

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    
    print("--- 1. Loading OOD Data ---")
    ood_url_path = project_root / "data/raw/OOD_URL.xlsx"
    ood_html_path = project_root / "data/raw/OOD_html.xlsx"

    df_ood_url = pd.read_excel(ood_url_path)
    df_ood_html = pd.read_excel(ood_html_path)

    df_ood_url['html'] = df_ood_html['Data']
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    print("--- 2. Extracting Features ---")
    processor = joblib.load(project_root / "data/processed/structural_rf/processor.joblib")
    X_ood = processor.transform(df_ood_url[['Data', 'html']])

    print("--- 3. Loading Base Pre-Trained Model ---")
    rf = joblib.load(project_root / "experiments/structural_rf/model.joblib")
    rf_probs = rf.predict_proba(X_ood)[:, 1].reshape(-1, 1)

    # Meta-features = [RF Probability, Raw Structural Features]
    meta_X = np.hstack((rf_probs, X_ood))

    sample_sizes = [10, 20, 50, 100, 200]
    n_seeds = 5
    
    results = []

    print("--- 4. Running Few-Shot Comparisons ---")
    for n_samples in sample_sizes:
        print(f"Evaluating with {n_samples} new zero-day samples...")
        
        acc_scratch_rf = []
        acc_meta_lr = []
        acc_meta_svm = []
        acc_knn = []
        
        for seed in range(n_seeds):
            # Split OOD into Few-Shot Train and Test
            X_train_few, X_test_few, y_train_few, y_test_few = train_test_split(
                meta_X, y_ood, train_size=n_samples, random_state=seed, stratify=y_ood
            )
            
            # The raw features are everything after the first column
            raw_X_train_few = X_train_few[:, 1:]
            raw_X_test_few = X_test_few[:, 1:]
            
            # --- Method 1: Baseline (RF from Scratch) ---
            # Training a brand new RF ONLY on the few new samples
            m_rf = RandomForestClassifier(n_estimators=50, random_state=seed)
            m_rf.fit(raw_X_train_few, y_train_few)
            acc_scratch_rf.append(accuracy_score(y_test_few, m_rf.predict(raw_X_test_few)))
            
            # --- Method 2: Transfer Learning (Meta-LR) ---
            m_lr = LogisticRegression(max_iter=1000, C=1.0)
            m_lr.fit(X_train_few, y_train_few)
            acc_meta_lr.append(accuracy_score(y_test_few, m_lr.predict(X_test_few)))
            
            # --- Method 3: Transfer Learning (Meta-SVM) ---
            m_svm = SVC(kernel='rbf', probability=False)
            m_svm.fit(X_train_few, y_train_few)
            acc_meta_svm.append(accuracy_score(y_test_few, m_svm.predict(X_test_few)))
            
            # --- Method 4: K-Nearest Neighbors (KNN) ---
            # Using only raw features, finding the nearest known sample
            n_neighbors = min(5, n_samples) # Ensure k <= samples
            m_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            m_knn.fit(raw_X_train_few, y_train_few)
            acc_knn.append(accuracy_score(y_test_few, m_knn.predict(raw_X_test_few)))

        # Average over seeds
        results.append({
            'N_Samples': n_samples,
            'Model': 'Baseline RF (Scratch)',
            'Accuracy': np.mean(acc_scratch_rf)
        })
        results.append({
            'N_Samples': n_samples,
            'Model': 'Transfer Learning (Meta-LR)',
            'Accuracy': np.mean(acc_meta_lr)
        })
        results.append({
            'N_Samples': n_samples,
            'Model': 'Transfer Learning (Meta-SVM)',
            'Accuracy': np.mean(acc_meta_svm)
        })
        results.append({
            'N_Samples': n_samples,
            'Model': 'K-Nearest Neighbors (KNN)',
            'Accuracy': np.mean(acc_knn)
        })

    df_res = pd.DataFrame(results)
    
    # Save Tabular Results
    os.makedirs(project_root / "docs", exist_ok=True)
    report_path = project_root / "docs/few_shot_report.md"
    
    pivot_df = df_res.pivot(index='N_Samples', columns='Model', values='Accuracy')
    
    with open(report_path, 'w') as f:
        f.write("# Few-Shot Learning Comparison Report\n\n")
        f.write("This report compares different machine learning architectures on their ability to reach the highest OOD accuracy using the absolute minimum amount of new data.\n\n")
        f.write("## Tabular Results (Zero-Shot Accuracy)\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n\n*Note: Zero-Shot accuracy (0 new samples) for the pre-trained RF is 68.8%.*\n")
        
    print(f"Saved tabular report to {report_path}")

    # Plot Learning Curves
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df_res, x='N_Samples', y='Accuracy', hue='Model', marker='o', linewidth=2.5, markersize=8)
    
    plt.axhline(0.688, color='k', linestyle='--', label='Original Pre-Trained RF (Zero-Shot)')
    
    plt.title('Few-Shot Adaptation Learning Curves (Zero-Day Phishing)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of New Data Points Used for Training', fontsize=12, fontweight='bold')
    plt.ylabel('OOD Accuracy', fontsize=12, fontweight='bold')
    plt.legend(title='Adaptation Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = project_root / "docs/assets/model_comparisons/few_shot_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved learning curve plot to {plot_path}")

if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

def main():
    print("--- Extracting Pruned Feature Statistics ---")
    proc_dir = Path("data/processed/structural_xgb_40k")
    
    print("Loading pre-processed data...")
    X = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y = joblib.load(proc_dir / "y_phresh_40k.joblib")
    processor = joblib.load(proc_dir / "processor.joblib")
    feature_names = processor.get_feature_names()
    
    print("Training full model to extract importances...")
    full_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    full_model.fit(X, y)
    
    importances = full_model.feature_importances_
    
    selector = SelectFromModel(full_model, threshold='median', prefit=True)
    kept_indices = selector.get_support(indices=True)
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Status': ['Retained' if i in kept_indices else 'Pruned' for i in range(len(feature_names))]
    })
    
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    # Plot top 20 retained and top 20 pruned
    top_retained = df_imp[df_imp['Status'] == 'Retained'].head(20)
    top_pruned = df_imp[df_imp['Status'] == 'Pruned'].head(20)
    plot_df = pd.concat([top_retained, top_pruned]).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(data=plot_df, x='Importance', y='Feature', hue='Status', palette={'Retained': '#2ca02c', 'Pruned': '#d62728'})
    plt.title('Feature Importance & RFE Pruning (Top 40 Displayed)')
    plt.tight_layout()
    
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / "feature_pruning_report.png"
    plt.savefig(out_path, dpi=300)
    print(f"Graph saved to {out_path}")
    
    # Save text log
    log_path = Path("docs/reports/pruned_features_list.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("# Feature Pruning Report\n\n")
        f.write("## Retained Features (Top 52)\n")
        for ftr in df_imp[df_imp['Status'] == 'Retained']['Feature']:
            f.write(f"- {ftr}\n")
        f.write("\n## Pruned Features (Noise Removed)\n")
        for ftr in df_imp[df_imp['Status'] == 'Pruned']['Feature']:
            f.write(f"- {ftr}\n")
            
    print(f"List saved to {log_path}")

if __name__ == "__main__":
    main()

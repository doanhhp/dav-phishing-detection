import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import xgboost as xgb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def load_data(dataset_name):
    path = project_root / 'data' / 'processed' / 'standardized' / f'{dataset_name.lower()}_dataset.parquet'
    df = pd.read_parquet(path)
    if 'Category' in df.columns:
        df = df.rename(columns={'Category': 'Class'})
    return df

def extract_features(df):
    from scripts.visualizations.plot_individual_features import extract_features as ef
    return ef(df)

def load_sampled_data(dataset_name):
    from scripts.visualizations.plot_individual_features import load_dataset
    # Returns 2000 samples (1000 Phishing, 1000 Legitimate)
    return load_dataset(dataset_name, samples_per_class=1000)

def main():
    out_dir = project_root / 'docs' / 'assets' / 'explainable_ai'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Main Dataset for training...")
    df_main = load_sampled_data('Main')
    df_features_main = extract_features(df_main)
    
    X_train = df_features_main.drop('Class', axis=1)
    # The 'Class' column contains strings 'Phishing' and 'Legitimate' because we used plot_individual_features.load_dataset
    y_train = (df_features_main['Class'] == 'Phishing').astype(int)
    
    print(f"Training XGBoost on Main... (Distribution: {y_train.sum()} Phishing out of {len(y_train)} rows)")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    # SHAP for Main
    print("Generating SHAP for Main Dataset...")
    shap_values_main = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_main, X_train, show=False, max_display=15)
    plt.title("SHAP Feature Importance (Trained and Tested on 2021 Main Dataset)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP for PhreshPhish
    print("Loading PhreshPhish Dataset...")
    df_phish = load_sampled_data('PhreshPhish')
    df_features_phish = extract_features(df_phish)
    
    X_test = df_features_phish.drop('Class', axis=1)
    
    print("Generating SHAP for PhreshPhish Dataset...")
    shap_values_phish = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_phish, X_test, show=False, max_display=15)
    plt.title("SHAP Feature Drift (Trained on Main, Tested on 2024 PhreshPhish)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_phreshphish.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP plots saved to {out_dir}")

if __name__ == "__main__":
    main()

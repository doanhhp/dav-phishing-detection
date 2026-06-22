import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from src.models.structural_xgb import Structural_XGB

def main():
    print("--- 1. Loading Mix of Old and New Data (20/80 Split) ---")
    df_old_url = pd.read_excel('data/raw/URL.xlsx').head(200)
    df_old_html = pd.read_excel('data/raw/html.xlsx').head(200)
    if 'Label' in df_old_url.columns:
        df_old_url = df_old_url.rename(columns={'Label': 'Category'})
    
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    df_new_train_url = df_new_url.iloc[:800].copy()
    df_new_train_html = df_new_html.iloc[:800].copy()
    
    df_train_url = pd.concat([df_old_url, df_new_train_url], ignore_index=True)
    df_train_html = pd.concat([df_old_html, df_new_train_html], ignore_index=True)
    
    y_train = df_train_url['Category'].map({'ham': 0, 'spam': 1}).values

    print("\n--- 2. Extracting Structural Features ---")
    from src.features.structural import StructuralProcessor
    struct_proc = StructuralProcessor(config={})
    
    df_train_raw = df_train_url[['Data']].copy()
    df_train_raw['html'] = df_train_html['Data']
    X_train = struct_proc.fit_transform(df_train_raw)

    feature_names = struct_proc.get_feature_names()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    print("\n--- 3. Training Incremental XGBoost ---")
    xgb_model = Structural_XGB(config={'n_estimators': 100, 'learning_rate': 0.1})
    xgb_model.fit(X_train, y_train)

    print("\n--- 4. Running SHAP Explainer ---")
    # Clean feature names to prevent XGBoost ValueError
    X_train_df.columns = [str(c).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for c in X_train_df.columns]

    # Using TreeExplainer which is optimized for XGBoost
    explainer = shap.TreeExplainer(xgb_model.model)
    shap_values = explainer.shap_values(X_train_df)

    out_dir = 'docs/assets/explainable_ai/xgboost'
    os.makedirs(out_dir, exist_ok=True)

    print("\n--- 5. Generating Visualizations ---")
    
    # 1. Summary Plot (Bar) - Global Feature Importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
    plt.title('XGBoost Feature Importance (Incremental 20/80 Mix)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'xgb_shap_summary_bar.png'))
    plt.close()
    
    # 2. Summary Plot (Dot) - Directional Impact
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train_df, show=False)
    plt.title('XGBoost SHAP Value Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'xgb_shap_summary_dot.png'))
    plt.close()

    print(f"SHAP plots successfully saved to {out_dir}")

if __name__ == "__main__":
    main()

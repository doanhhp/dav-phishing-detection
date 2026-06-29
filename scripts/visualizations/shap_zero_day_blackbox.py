import pandas as pd
import numpy as np
import joblib
import os
import sys
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from src.features.structural import StructuralProcessor

def main():
    print("Loading 2026 Zero-Day Data (Legitimate)...")
    df_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_2026['html'] = df_html_2026['Data']
    
    # Isolate False Positives (Legitimate sites that the model will misclassify)
    df_2026_legit = df_2026[df_2026['Category'].isin(['ham', 'benign'])].copy()
    
    # Sample 50 sites for KernelExplainer since it's computationally heavy
    df_sample = df_2026_legit.sample(50, random_state=42)
    
    print("\nLoading Processor and Model...")
    processor = joblib.load("data/processed/mid_fusion_xgb/processor.joblib")
    
    from src.models.model_factory import ModelFactory
    import yaml
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = ModelFactory.create_model("mid_fusion_xgb", config["models"]["mid_fusion_xgb"])
    model.load("experiments/mid_fusion_xgb/model")
    
    print("Processing Features...")
    X_processed = processor.transform(df_sample)
    
    feature_names = processor.get_feature_names()
    
    print("Running SHAP KernelExplainer (Blackbox Mode)...")
    # Define a prediction function that outputs the probability of class 1 (Phishing)
    def predict_fn(X):
        return model.predict_proba(X)[:, 1]
    
    # We use a small background dataset to initialize the explainer (e.g. zeros, or a mean vector)
    # Actually, we can use K-Means summary of the 2021 legitimate data as background
    # But for speed, let's just use the median of the 2026 sample
    background = shap.kmeans(X_processed, 5)
    
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_processed)
    
    # Generate SHAP Summary Plot
    out_dir = "docs/assets/explainable_ai/zero_day_failure"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
    plt.title("SHAP Explainability: Why 2026 Legitimate Sites were Flagged as Phishing", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_fp_summary.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP analysis complete! Image saved to {out_dir}/shap_fp_summary.png")

if __name__ == '__main__':
    main()

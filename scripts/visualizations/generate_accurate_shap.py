import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import xgboost as xgb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from scripts.visualizations.plot_individual_features import load_dataset
from src.features.structural import StructuralProcessor

def main():
    out_dir = project_root / 'docs' / 'assets' / 'explainable_ai'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets (Full Dataset)...")
    # Using the full dataset as requested by the user
    df_main = load_dataset('Main')
    df_phish = load_dataset('PhreshPhish')
    
    print("Loading exact models and processors used for Structural XGBoost...")
    model_path = project_root / 'experiments' / 'structural_xgb' / 'model.joblib'
    proc_path = project_root / 'data' / 'processed' / 'structural_xgb' / 'processor.joblib'
    
    # Check if we should use structural_xgb or mid_fusion_xgb processor
    # structural_xgb is safer because it explicitly evaluates the 163 DOM features directly
    
    xgb_model = joblib.load(model_path)
    if hasattr(xgb_model, 'best_estimator_'):
        xgb_model = xgb_model.best_estimator_
        
    processor = joblib.load(proc_path)
    if not hasattr(processor, 'upper_bounds'):
        processor.upper_bounds = None
    
    print("Transforming Main Dataset (this will extract all 163 DOM features, including TF-IDF)...")
    X_main = processor.transform(df_main)
    feature_names = processor.get_feature_names()
    
    # XGBoost doesn't allow < or > in feature names, and the model was trained on numpy arrays anyway
    safe_feature_names = [f.replace('<', '').replace('>', '').replace('[', '').replace(']', '') for f in feature_names]
    
    print("Generating SHAP values for Main Dataset...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_main = explainer.shap_values(X_main)
    
    plt.figure(figsize=(10, 8))
    # cmap="coolwarm" to clearly show High vs Low
    shap.summary_plot(shap_values_main, X_main, feature_names=safe_feature_names, show=False, max_display=15)
    plt.title("Accurate SHAP Feature Importance (Dataset-2021)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_summary_main.png")
    
    print("Transforming PhreshPhish Dataset...")
    X_phish = processor.transform(df_phish)
    
    print("Generating SHAP values for PhreshPhish Dataset...")
    shap_values_phish = explainer.shap_values(X_phish)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_phish, X_phish, feature_names=safe_feature_names, show=False, max_display=15)
    plt.title("Accurate SHAP Feature Drift (Tested on Dataset-2024 PhreshPhish)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_phreshphish.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_summary_phreshphish.png")
    print("Transforming OOD Dataset...")
    try:
        df_ood = load_dataset('OOD')
        X_ood = processor.transform(df_ood)
        
        print("Generating SHAP values for OOD Dataset...")
        shap_values_ood = explainer.shap_values(X_ood)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_ood, X_ood, feature_names=safe_feature_names, show=False, max_display=15)
        plt.title("Accurate SHAP Feature Drift (Tested on Dataset-2026 OOD)")
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_summary_ood.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved shap_summary_ood.png")
    except Exception as e:
        print(f"Skipping OOD dataset: {e}")

if __name__ == "__main__":
    main()

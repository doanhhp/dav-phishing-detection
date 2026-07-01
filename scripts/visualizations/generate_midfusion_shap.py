import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from scripts.visualizations.plot_individual_features import load_dataset
from src.features.structural import StructuralProcessor
from src.models.mid_fusion_xgb import MidFusionXGB

def main():
    out_dir = project_root / 'docs' / 'assets' / 'explainable_ai'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    # Sample 1000 Phishing and 1000 Legitimate to make SHAP fast
    df_main = load_dataset('Main', samples_per_class=1000)
    df_phish = load_dataset('PhreshPhish', samples_per_class=1000)
    
    print("Loading exact models and processors used for Mid-Fusion XGBoost...")
    model_path = project_root / 'experiments' / 'mid_fusion_xgb' / 'model.joblib'
    proc_path = project_root / 'data' / 'processed' / 'mid_fusion_xgb' / 'processor.joblib'
    
    # Load StackingClassifier from the MidFusionXGB wrapper
    model_wrapper = MidFusionXGB({})
    model_wrapper.load(str(model_path).replace('.joblib', '')) 
    stacking_clf = model_wrapper.model
    
    processor = joblib.load(proc_path)
    if not hasattr(processor, 'upper_bounds'):
        processor.upper_bounds = None
    
    feature_names = processor.get_feature_names()
    safe_feature_names = [f.replace('<', '').replace('>', '').replace('[', '').replace(']', '') for f in feature_names]
    
    # The meta-features for Mid-Fusion are [URL_Expert_Prob, HTML_Expert_Prob] + [All Raw Features]
    meta_feature_names = ['URL_Expert_Prob', 'HTML_Expert_Prob'] + safe_feature_names
    
    print("Transforming Main Dataset (this will extract all 163 DOM features)...")
    X_main_raw = processor.transform(df_main)
    print("Generating Meta-Matrix via StackingClassifier...")
    X_main_meta = stacking_clf.transform(X_main_raw)
    
    print("Generating SHAP values for Main Dataset (Mid-Fusion Meta-Learner)...")
    joint_expert = stacking_clf.final_estimator_
    explainer = shap.TreeExplainer(joint_expert)
    shap_values_main = explainer.shap_values(X_main_meta)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_main, X_main_meta, feature_names=meta_feature_names, show=False, max_display=15)
    plt.title("Mid-Fusion SHAP Feature Importance (Dataset-2021)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_midfusion_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_midfusion_main.png")
    
    print("Transforming PhreshPhish Dataset...")
    X_phish_raw = processor.transform(df_phish)
    X_phish_meta = stacking_clf.transform(X_phish_raw)
    
    print("Generating SHAP values for PhreshPhish Dataset (Mid-Fusion Meta-Learner)...")
    shap_values_phish = explainer.shap_values(X_phish_meta)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_phish, X_phish_meta, feature_names=meta_feature_names, show=False, max_display=15)
    plt.title("Mid-Fusion SHAP Feature Drift (Dataset-2024 PhreshPhish)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_midfusion_phreshphish.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_midfusion_phreshphish.png")
    print("Transforming OOD Dataset (Dataset-2026)...")
    try:
        df_ood = load_dataset('OOD', samples_per_class=1000)
        X_ood_raw = processor.transform(df_ood)
        X_ood_meta = stacking_clf.transform(X_ood_raw)
        
        print("Generating SHAP values for OOD Dataset (Mid-Fusion Meta-Learner)...")
        shap_values_ood = explainer.shap_values(X_ood_meta)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_ood, X_ood_meta, feature_names=meta_feature_names, show=False, max_display=15)
        plt.title("Mid-Fusion SHAP Feature Drift (Dataset-2026 OOD)")
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_midfusion_ood.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved shap_midfusion_ood.png")
    except Exception as e:
        print(f"Skipping OOD dataset: {e}")

if __name__ == "__main__":
    main()

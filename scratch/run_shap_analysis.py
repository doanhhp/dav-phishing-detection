import os
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from src.utils.config_loader import ConfigLoader

project_root = Path(__file__).resolve().parent.parent

def run_shap_for_model(model_name, X_train, X_test, feature_names, out_dir):
    print(f"\n[{model_name}] Training model and generating SHAP values...")
    
    # Load config and initialize model
    config = ConfigLoader.load_yaml(project_root / 'config' / 'benchmarks.yaml')
    model = ModelFactory.create_model(model_name, config['models'][model_name])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Initialize TreeExplainer (fastest for Random Forest / XGBoost)
    if hasattr(model.model, 'estimators_') or 'xgb' in model_name: # Random Forest or XGBoost
        # Sample background for speed if needed, but TreeExplainer handles full sets fast
        explainer = shap.TreeExplainer(model.model)
        # We'll use a subset of test data for SHAP to save time (e.g., 500 samples)
        X_test_sample = X_test[:500] if len(X_test) > 500 else X_test
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_sample)
        
        # Handle multi-class output format in newer SHAP/scikit-learn
        if isinstance(shap_values, list):
            # For binary classification, shap_values[1] is the positive class (Phishing)
            shap_values_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values
            
        print(f"[{model_name}] SHAP values computed. Generating plots...")
        
        # Plot 1: Summary Plot (Beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, X_test_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary (Feature Impact) - {model_name}', y=1.05)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Bar Plot (Global Importance)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, X_test_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Global Importance - {model_name}', y=1.05)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Dependence Plot (Top Feature)
        # Get mean absolute SHAP values to find the top feature
        mean_shap = np.abs(shap_values_to_plot).mean(axis=0)
        top_feature_idx = np.argmax(mean_shap)
        top_feature_name = feature_names[top_feature_idx]
        
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(top_feature_idx, shap_values_to_plot, X_test_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot: {top_feature_name}', y=1.05)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Local Force Plot (First Phishing Prediction)
        # Find the first phishing site in the sample
        predictions = model.predict(X_test_sample)
        phishing_indices = np.where(predictions == 1)[0]
        
        if len(phishing_indices) > 0:
            idx = phishing_indices[0] # Pick the first phishing site
            # We must use matplotlib for force plot to save it as PNG easily
            plt.figure(figsize=(15, 4))
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                shap_values_to_plot[idx,:], 
                X_test_sample[idx,:], 
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"Local Explanation (Single Phishing Site) - {model_name}")
            plt.savefig(out_dir / 'shap_local_force.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall Plot (Newer, cleaner local plot)
            plt.figure(figsize=(10, 6))
            # Create Explanation object for waterfall
            exp = shap.Explanation(
                values=shap_values_to_plot[idx,:], 
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                data=X_test_sample[idx,:], 
                feature_names=feature_names
            )
            shap.waterfall_plot(exp, show=False)
            plt.title(f"Local Explanation (Waterfall) - {model_name}")
            plt.tight_layout()
            plt.savefig(out_dir / 'shap_local_waterfall.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"[{model_name}] Completed.")
    else:
        print(f"[{model_name}] SHAP TreeExplainer not supported for this model type. Skipping.")


if __name__ == "__main__":
    print("--- 1. Loading Datasets ---")
    
    try:
        df_url = pd.read_excel(project_root / 'data/raw/URL.xlsx', engine='calamine').sample(3000, random_state=42)
        df_html = pd.read_excel(project_root / 'data/raw/html.xlsx', engine='calamine')
    except:
        df_url = pd.read_excel(project_root / 'data/raw/URL.xlsx').sample(3000, random_state=42)
        df_html = pd.read_excel(project_root / 'data/raw/html.xlsx')
        
    df_html = df_html.loc[df_url.index] # Match samples
    
    config = ConfigLoader.load_yaml(project_root / 'config' / 'benchmarks.yaml')
    
    models_to_run = ['structural_rf', 'url_rf', 'structural_xgb']
    
    for model_name in models_to_run:
        out_dir = project_root / 'docs' / 'assets' / 'explainable_ai' / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Get processor type for the model
        processor_type = config['models'][model_name].get('processor', 'structural')
        
        # 2. Extract features
        print(f"\n--- Extracting Features for {model_name} ({processor_type}) ---")
        processor = FeatureFactory.get_processor(processor_type, config['features'])
        
        # Handle fit_transform properly
        df_combined = pd.DataFrame({'Data': df_url['Data'], 'html': df_html['Data']})
        
        if processor_type == 'structural':
            X = processor.fit_transform(df_combined)
        elif processor_type == 'url':
            X = processor.fit_transform(df_url[['Data']])
        else:
            continue
            
        global y_train # To share with train function simply
        y = (df_url['Category'] == 'spam').astype(int).values
        
        # Split train/test (we'll use a simple manual split here to keep X_test as numpy array for SHAP)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        feature_names = processor.get_feature_names()
        
        # 3. Run SHAP
        run_shap_for_model(model_name, X_train, X_test, feature_names, out_dir)

    print("\nAll SHAP Analyses Complete! Visualizations saved in docs/assets/explainable_ai/")

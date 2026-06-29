import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

def main():
    print("Loading 2021 Data (Legitimate)...")
    df_2021 = pd.read_excel("data/raw/URL.xlsx")
    df_html_2021 = pd.read_excel("data/raw/html.xlsx")
    
    if 'html' in df_html_2021.columns:
        df_2021['html'] = df_html_2021['html']
    else:
        df_2021['html'] = df_html_2021['Data']
        
    df_2021_legit = df_2021[df_2021['Category'].isin(['ham', 'benign'])]
    print(f"2021 Legitimate Sites: {len(df_2021_legit)}")
    
    print("Loading 2026 Data (Legitimate)...")
    df_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_2026['html'] = df_html_2026['Data']
    
    df_2026_legit = df_2026[df_2026['Category'].isin(['ham', 'benign'])]
    print(f"2026 Legitimate Sites: {len(df_2026_legit)}")
    
    print("\nLoading Feature Processor...")
    try:
        processor = joblib.load("data/processed/mid_fusion_xgb/processor.joblib")
    except:
        from src.features.structural import StructuralProcessor
        import yaml
        with open("config/benchmarks.yaml", "r") as f:
            config = yaml.safe_load(f)
        processor = StructuralProcessor(config)
        processor.fit_transform(df_2021.sample(100)) # Just to initialize
        
    print("Extracting Features for 2021 Legit...")
    # Just take a sample to speed it up
    df_2021_sample = df_2021_legit.sample(1000, random_state=42)
    feat_2021 = []
    for _, row in df_2021_sample.iterrows():
        from urllib.parse import urlparse
        u = str(row['Data'])
        h = str(row['html'])
        domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
        feat_2021.append(processor._extract_numerical_features(u, h, domain))
        
    print("Extracting Features for 2026 Legit...")
    feat_2026 = []
    for _, row in df_2026_legit.iterrows():
        from urllib.parse import urlparse
        u = str(row['Data'])
        h = str(row['html'])
        domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
        feat_2026.append(processor._extract_numerical_features(u, h, domain))
        
    feat_2021 = np.array(feat_2021)
    feat_2026 = np.array(feat_2026)
    
    # The first 28 features from get_feature_names()
    feature_names = [
        'url_length', 'url_num_dots', 'is_https', 'url_num_special_chars', 'url_digit_ratio',
        'url_num_subdomains', 'url_entropy', 'url_path_depth', 'url_has_login', 'url_hyphen_domain',
        'html_length', 'html_num_tags', 'html_text_ratio', 'html_script_count',
        'html_external_link_ratio', 'html_password_input_count',
        'html_empty_link_ratio', 'html_input_to_p_ratio', 'css_hidden_count',
        'html_title_length', 'brand_discrepancy', 'dom_depth', 'tag_diversity', 'external_resource_ratio', 'foreign_form_action',
        'html_js_ratio', 'body_tag_count', 'iframe_count'
    ]
    
    print("\n--- MAJOR STRUCTURAL SHIFTS IN LEGITIMATE SITES (2021 vs 2026) ---")
    print(f"{'Feature':<30} | {'2021 Avg':<15} | {'2026 Avg':<15} | {'% Change':<10}")
    print("-" * 75)
    
    for i, name in enumerate(feature_names):
        avg_21 = np.mean(feat_2021[:, i])
        avg_26 = np.mean(feat_2026[:, i])
        
        if avg_21 == 0:
            pct_change = float('inf') if avg_26 > 0 else 0
        else:
            pct_change = ((avg_26 - avg_21) / avg_21) * 100
            
        if abs(pct_change) > 50: # Only show massive shifts
            print(f"{name:<30} | {avg_21:<15.4f} | {avg_26:<15.4f} | {pct_change:+.1f}%")

if __name__ == '__main__':
    main()

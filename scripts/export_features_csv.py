import pandas as pd
from src.features.structural import StructuralProcessor

def export_preview():
    print("Loading data...")
    df_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    df_combined = df_url.copy()
    df_combined['html'] = df_html['Data']
    
    # Process features
    print("Extracting features...")
    processor = StructuralProcessor({})
    # We shouldn't use scale here so the user can see actual values (e.g. dom depth = 14) instead of scaled values.
    # So I will temporarily disable the scaler logic or just run extract_numerical manually
    
    features = []
    from urllib.parse import urlparse
    for u, h in zip(df_combined['Data'], df_combined['html']):
        domain = urlparse(u if str(u).startswith('http') else f"http://{u}").netloc
        features.append(processor._extract_numerical_features(str(u), str(h), domain))
        
    feature_names = [
        'url_length', 'url_num_dots', 'url_num_special_chars', 'url_digit_ratio',
        'url_num_subdomains', 'url_entropy', 'url_path_depth', 'url_has_login', 'url_hyphen_domain',
        'html_length', 'html_num_tags', 'html_text_ratio', 'html_script_count',
        'html_external_link_ratio', 'html_password_input_count',
        'html_empty_link_ratio', 'html_input_to_p_ratio', 'css_hidden_count',
        'html_title_length', 'brand_discrepancy', 'dom_depth', 'tag_diversity', 'external_resource_ratio', 'foreign_form_action'
    ]
    
    df_features = pd.DataFrame(features, columns=feature_names)
    df_features.insert(0, 'URL', df_combined['Data'])
    df_features.insert(1, 'Category', df_combined['Category'])
    
    out_path = 'data/processed/structural_features_preview.csv'
    df_features.to_csv(out_path, index=False)
    print(f"Saved {len(df_features)} rows to {out_path}")

if __name__ == "__main__":
    export_preview()

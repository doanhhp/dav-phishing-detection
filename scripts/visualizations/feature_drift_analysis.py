import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Datasets ---")
    # Load historical
    df_old_url = pd.read_excel('data/raw/URL.xlsx')
    df_old_html = pd.read_excel('data/raw/html.xlsx')
    
    # Load new
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    # We only care about phishing websites for this drift analysis (how have scammers changed?)
    # Old phishing
    old_spam_idx = df_old_url['Label'] == 'spam' if 'Label' in df_old_url.columns else df_old_url['Category'] == 'spam'
    df_old_phish_url = df_old_url[old_spam_idx].head(2000)
    df_old_phish_html = df_old_html[old_spam_idx].head(2000)
    
    # New phishing
    new_spam_idx = df_new_url['Category'] == 'spam'
    df_new_phish_url = df_new_url[new_spam_idx].head(2000)
    df_new_phish_html = df_new_html[new_spam_idx].head(2000)
    
    print("--- 2. Extracting Structural Features ---")
    proc = StructuralProcessor({})
    
    # Old Features
    df_old = df_old_phish_url[['Data']].copy()
    df_old['html'] = df_old_phish_html['Data'].values
    X_old = proc.fit_transform(df_old)
    
    # New Features
    df_new = df_new_phish_url[['Data']].copy()
    df_new['html'] = df_new_phish_html['Data'].values
    X_new = proc.transform(df_new)
    
    feature_names = proc.get_feature_names()
    
    df_old_feat = pd.DataFrame(X_old, columns=feature_names)
    df_old_feat['Era'] = '2021 Phishing'
    
    df_new_feat = pd.DataFrame(X_new, columns=feature_names)
    df_new_feat['Era'] = '2026 Phishing'
    
    df_combined = pd.concat([df_old_feat, df_new_feat], ignore_index=True)
    
    print("--- 3. Plotting Feature Drift ---")
    # Select a mix of URL and HTML features to plot
    features_to_plot = [
        'url_length', 'url_entropy', 'dom_depth', 
        'css_hidden_count', 'html_text_ratio', 'foreign_form_action',
        'tag_diversity', 'html_external_link_ratio', 'html_password_input_count'
    ]
    
    # Filter to features that actually exist in the model
    features_to_plot = [f for f in features_to_plot if f in feature_names]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        if i >= len(axes): break
        
        # Calculate Population Stability Index (visual approximation via KDE)
        sns.kdeplot(data=df_combined, x=feature, hue='Era', fill=True, ax=axes[i], common_norm=False, palette={'2021 Phishing': 'red', '2026 Phishing': 'blue'}, alpha=0.5)
        
        axes[i].set_title(f'Drift Analysis: {feature}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Feature Value')
        axes[i].set_ylabel('Density')
        
    plt.tight_layout()
    output_path = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_drift_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved drift analysis plot to: {output_path}")

if __name__ == "__main__":
    main()

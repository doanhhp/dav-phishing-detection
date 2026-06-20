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
    # We will use the modern 2026 OOD dataset to show why these features matter TODAY
    df_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    if 'Label' in df_url.columns:
        df_url = df_url.rename(columns={'Label': 'Category'})
        
    print("--- 2. Extracting Structural Features ---")
    proc = StructuralProcessor({})
    
    df_raw = df_url[['Data']].copy()
    df_raw['html'] = df_html['Data'].values
    
    # We want raw, unscaled features so the x-axis makes sense to a human
    # fit_transform usually applies StandardScaler. We can inverse_transform if needed,
    # or just extract manually. Let's just use the transformed features, the separation is what matters.
    # Actually, to make it easier for a lecturer to read, let's turn off scaling if possible,
    # or just accept the scaled Z-scores. Z-scores are fine for showing separation.
    
    X = proc.fit_transform(df_raw)
    feature_names = proc.get_feature_names()
    
    df_feat = pd.DataFrame(X, columns=feature_names)
    df_feat['Class'] = df_url['Category'].map({'ham': 'Legitimate (Ham)', 'spam': 'Phishing (Spam)'})
    
    print("--- 3. Plotting Feature Justification ---")
    # Choose 6 highly predictive features to explain to the lecturer
    features_to_plot = [
        'url_length', 'url_entropy', 'dom_depth', 
        'css_hidden_count', 'html_text_ratio', 'html_password_input_count'
    ]
    
    # Filter to ensure they exist
    features_to_plot = [f for f in features_to_plot if f in feature_names]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    palette = {'Legitimate (Ham)': 'green', 'Phishing (Spam)': 'red'}
    
    for i, feature in enumerate(features_to_plot):
        if i >= len(axes): break
        
        sns.kdeplot(data=df_feat, x=feature, hue='Class', fill=True, ax=axes[i], common_norm=False, palette=palette, alpha=0.5)
        
        # Make titles intuitive for the presentation
        readable_title = feature.replace('_', ' ').title()
        axes[i].set_title(f'Why we chose: {readable_title}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Mathematical Score (Z-Normalized)')
        axes[i].set_ylabel('Density of Websites')
        
    plt.tight_layout()
    output_path = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_justification.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature justification plot to: {output_path}")

if __name__ == "__main__":
    main()

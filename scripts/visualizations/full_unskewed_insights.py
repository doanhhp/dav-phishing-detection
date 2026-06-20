import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from src.features.structural import StructuralProcessor
import warnings
warnings.filterwarnings('ignore')

def extract_raw_features_full(df_url, df_html):
    """Extracts raw numerical features for the ENTIRE dataset without downsampling."""
    proc = StructuralProcessor({})
    base_features = proc.get_feature_names()[:26]
    
    urls = df_url['Data'].astype(str).values
    htmls = df_html['Data'].astype(str).values
    categories = df_url['Label' if 'Label' in df_url.columns else 'Category'].values
    
    data = []
    for u, h, cat in zip(urls, htmls, categories):
        domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
        feats = proc._extract_numerical_features(u, h, domain)
        row = dict(zip(base_features, feats))
        row['Class'] = 'Phishing' if cat == 'spam' else 'Legitimate'
        data.append(row)
        
    return pd.DataFrame(data), base_features

def plot_unskewed_proportions(df, features, output_path, title):
    """Plots binary features as a percentage. This natively handles class imbalance."""
    valid_features = [f for f in features if f in df.columns]
    
    prop_data = []
    for f in valid_features:
        for c in ['Legitimate', 'Phishing']:
            val_mean = (df[df['Class'] == c][f] > 0).mean() * 100 # Percentage!
            prop_data.append({'Feature': f, 'Class': c, 'Percentage': val_mean})
            
    df_prop = pd.DataFrame(prop_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_prop, x='Percentage', y='Feature', hue='Class', palette={'Legitimate': '#4CAF50', 'Phishing': '#F44336'})
    plt.title(f"{title}\n(Percentage of sites - Imbalance Neutralized)", fontsize=16, fontweight='bold')
    plt.xlabel("Percentage (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_unskewed_averages(df, features, output_path, title):
    """Plots the AVERAGE value for continuous features. Means are independent of sample size."""
    valid_features = [f for f in features if f in df.columns]
    
    mean_data = []
    for f in valid_features:
        for c in ['Legitimate', 'Phishing']:
            val_mean = df[df['Class'] == c][f].mean()
            mean_data.append({'Feature': f, 'Class': c, 'Average Value': val_mean})
            
    df_mean = pd.DataFrame(mean_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_mean, x='Average Value', y='Feature', hue='Class', palette={'Legitimate': '#4CAF50', 'Phishing': '#F44336'})
    plt.title(f"{title}\n(Average Values per Site - Imbalance Neutralized)", fontsize=16, fontweight='bold')
    plt.xlabel("Average Mathematical Value")
    plt.ylabel("")
    
    # Log scale x-axis since some averages (like length) are massive
    plt.xscale('log')
    plt.xlabel("Average Mathematical Value (Log Scale)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_unskewed_kde(df, features, output_path, title):
    """Plots KDE distributions using common_norm=False to normalize areas to 1, negating imbalance."""
    valid_features = [f for f in features if f in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(valid_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    axes = axes.flatten()
    
    for i, f in enumerate(valid_features):
        p95 = df[f].quantile(0.95)
        df_clean = df[df[f] <= p95 * 2] if p95 > 0 else df
        
        # common_norm=False normalizes each class curve independently!
        sns.kdeplot(data=df_clean, x=f, hue='Class', fill=True, ax=axes[i], palette={'Legitimate': 'green', 'Phishing': 'red'}, alpha=0.5, common_norm=False)
        axes[i].set_title(f, fontweight='bold')
        axes[i].set_ylabel("Density (Normalized)")
        
    for i in range(len(valid_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(title, fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def generate_full_unskewed_visuals(df, out_dir, year):
    binary_feats = ['is_https', 'url_has_login', 'url_hyphen_domain', 'foreign_form_action']
    continuous_feats = ['html_length', 'url_length', 'html_title_length', 'dom_depth', 'tag_diversity', 'html_num_tags', 'css_hidden_count']
    kde_feats = ['url_entropy', 'url_digit_ratio', 'html_text_ratio', 'html_external_link_ratio', 'html_empty_link_ratio']
    
    plot_unskewed_proportions(df, binary_feats, os.path.join(out_dir, f'unskewed_1_binary_proportions_{year}.png'), f"{year}: Binary Features")
    plot_unskewed_averages(df, continuous_feats, os.path.join(out_dir, f'unskewed_2_continuous_averages_{year}.png'), f"{year}: Continuous Features")
    plot_unskewed_kde(df, kde_feats, os.path.join(out_dir, f'unskewed_3_distribution_kde_{year}.png'), f"{year}: Normalized Density Distributions")

def main():
    print("--- 1. Loading FULL Datasets (No Sampling) ---")
    try:
        df_old_url = pd.read_excel('data/raw/URL.xlsx')
        df_old_html = pd.read_excel('data/raw/html.xlsx')
    except:
        df_old_url, df_old_html = None, None
        
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    out_dir_old = r'd:\Desktop\PhishingDetection\docs\assets\feature_insights_old'
    out_dir_new = r'd:\Desktop\PhishingDetection\docs\assets\feature_insights_new'
    os.makedirs(out_dir_old, exist_ok=True)
    os.makedirs(out_dir_new, exist_ok=True)
    
    if df_old_url is not None:
        print("--- 2. Extracting FULL 2021 Data ---")
        df_feat_old, _ = extract_raw_features_full(df_old_url, df_old_html)
        generate_full_unskewed_visuals(df_feat_old, out_dir_old, 2021)
        
    print("--- 3. Extracting FULL 2026 Data ---")
    df_feat_new, _ = extract_raw_features_full(df_new_url, df_new_html)
    generate_full_unskewed_visuals(df_feat_new, out_dir_new, 2026)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()

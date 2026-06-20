import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from src.features.structural import StructuralProcessor

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def extract_raw_features(df_url, df_html, n_samples=2000):
    df_url_sample = df_url.sample(min(n_samples, len(df_url)), random_state=42)
    df_html_sample = df_html.loc[df_url_sample.index]
    
    proc = StructuralProcessor({})
    base_features = proc.get_feature_names()[:26]
    
    urls = df_url_sample['Data'].astype(str).values
    htmls = df_html_sample['Data'].astype(str).values
    categories = df_url_sample['Label' if 'Label' in df_url_sample.columns else 'Category'].values
    
    data = []
    for u, h, cat in zip(urls, htmls, categories):
        domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
        feats = proc._extract_numerical_features(u, h, domain)
        row = dict(zip(base_features, feats))
        row['Class'] = 'Phishing' if cat == 'spam' else 'Legitimate'
        data.append(row)
        
    return pd.DataFrame(data), base_features

def plot_binary_features(df, features, output_path, title):
    valid_features = [f for f in features if f in df.columns]
    if not valid_features: return
    
    # Calculate proportions
    prop_data = []
    for f in valid_features:
        for c in ['Legitimate', 'Phishing']:
            # Force binary
            val_mean = (df[df['Class'] == c][f] > 0).mean() * 100
            prop_data.append({'Feature': f, 'Class': c, 'Percentage': val_mean})
            
    df_prop = pd.DataFrame(prop_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_prop, x='Percentage', y='Feature', hue='Class', palette={'Legitimate': 'lightgreen', 'Phishing': 'lightcoral'})
    plt.title(f"{title}\n(Percentage of sites exhibiting the trait)", fontsize=16, fontweight='bold')
    plt.xlabel("Percentage (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_continuous_log(df, features, output_path, title):
    valid_features = [f for f in features if f in df.columns]
    if not valid_features: return
    
    fig, axes = plt.subplots(1, len(valid_features), figsize=(18, 6))
    if len(valid_features) == 1: axes = [axes]
    
    for i, f in enumerate(valid_features):
        df_copy = df.copy()
        df_copy[f] = np.log1p(df_copy[f].clip(lower=0)) # Log scale for massive numbers
        sns.violinplot(data=df_copy, x='Class', y=f, ax=axes[i], palette={'Legitimate': 'lightgreen', 'Phishing': 'lightcoral'}, inner='quartile')
        axes[i].set_title(f"{f} (Log Scale)", fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Log Value")
        
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_kde_distributions(df, features, output_path, title):
    valid_features = [f for f in features if f in df.columns]
    if not valid_features: return
    
    n_cols = 2
    n_rows = int(np.ceil(len(valid_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
    axes = axes.flatten()
    
    for i, f in enumerate(valid_features):
        # Clip top 5% extreme outliers to make KDE readable
        p95 = df[f].quantile(0.95)
        df_clean = df[df[f] <= p95 * 1.5] if p95 > 0 else df
        
        sns.kdeplot(data=df_clean, x=f, hue='Class', fill=True, ax=axes[i], palette={'Legitimate': 'green', 'Phishing': 'red'}, alpha=0.5, common_norm=False)
        axes[i].set_title(f, fontweight='bold')
        
    for i in range(len(valid_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_ratio_histograms(df, features, output_path, title):
    valid_features = [f for f in features if f in df.columns]
    if not valid_features: return
    
    n_cols = 3
    n_rows = int(np.ceil(len(valid_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    
    for i, f in enumerate(valid_features):
        sns.histplot(data=df, x=f, hue='Class', ax=axes[i], palette={'Legitimate': 'green', 'Phishing': 'red'}, bins=20, multiple="dodge")
        axes[i].set_title(f, fontweight='bold')
        
    for i in range(len(valid_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_count_boxen(df, features, output_path, title):
    valid_features = [f for f in features if f in df.columns]
    if not valid_features: return
    
    n_cols = 4
    n_rows = int(np.ceil(len(valid_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    axes = axes.flatten()
    
    for i, f in enumerate(valid_features):
        # Clip top 5% extreme outliers to make Boxen readable
        p95 = df[f].quantile(0.95)
        df_clean = df[df[f] <= p95 * 2] if p95 > 0 else df
        
        sns.boxenplot(data=df_clean, x='Class', y=f, ax=axes[i], palette={'Legitimate': 'lightgreen', 'Phishing': 'lightcoral'})
        axes[i].set_title(f, fontweight='bold')
        axes[i].set_xlabel("")
        
    for i in range(len(valid_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def generate_visuals(df, out_dir, year):
    binary_feats = ['is_https', 'url_has_login', 'url_hyphen_domain', 'known_brand_mimicry', 'foreign_form_action']
    log_cont_feats = ['html_length', 'url_length', 'html_title_length']
    kde_feats = ['url_entropy', 'dom_depth', 'tag_diversity', 'html_num_tags']
    ratio_feats = ['url_digit_ratio', 'html_text_ratio', 'html_external_link_ratio', 'html_empty_link_ratio', 'external_resource_ratio', 'brand_discrepancy']
    count_feats = ['url_num_dots', 'url_num_special_chars', 'url_num_subdomains', 'url_path_depth', 'html_script_count', 'html_password_input_count', 'css_hidden_count', 'html_input_to_p_ratio']
    
    plot_binary_features(df, binary_feats, os.path.join(out_dir, f'1_binary_proportions_{year}.png'), f"{year}: Binary Features (Proportion of True)")
    plot_continuous_log(df, log_cont_feats, os.path.join(out_dir, f'2_continuous_violins_{year}.png'), f"{year}: Massive Distributions (Violin + Log Scale)")
    plot_kde_distributions(df, kde_feats, os.path.join(out_dir, f'3_distribution_kde_{year}.png'), f"{year}: Normal Distributions (KDE Density)")
    plot_ratio_histograms(df, ratio_feats, os.path.join(out_dir, f'4_ratio_histograms_{year}.png'), f"{year}: Ratio Features (Histograms 0-1)")
    plot_count_boxen(df, count_feats, os.path.join(out_dir, f'5_count_boxen_{year}.png'), f"{year}: Count Features (Boxen Plots for Outliers)")

def main():
    print("--- 1. Loading Datasets ---")
    try:
        df_old_url = pd.read_excel('data/raw/URL.xlsx')
        df_old_html = pd.read_excel('data/raw/html.xlsx')
    except Exception as e:
        df_old_url, df_old_html = None, None
        
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    out_dir_old = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_insights_old'
    out_dir_new = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_insights_new'
    os.makedirs(out_dir_old, exist_ok=True)
    os.makedirs(out_dir_new, exist_ok=True)
    
    if df_old_url is not None:
        print("--- 2. Generating Tailored Charts (2021 Data) ---")
        df_feat_old, _ = extract_raw_features(df_old_url, df_old_html, n_samples=2000)
        generate_visuals(df_feat_old, out_dir_old, 2021)
        
    print("--- 3. Generating Tailored Charts (2026 Data) ---")
    df_feat_new, _ = extract_raw_features(df_new_url, df_new_html, n_samples=2000)
    generate_visuals(df_feat_new, out_dir_new, 2026)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()

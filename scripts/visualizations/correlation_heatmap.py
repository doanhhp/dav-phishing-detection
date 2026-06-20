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

def extract_features_for_corr(df_url, df_html, n_samples=2000):
    df_url_sample = df_url.sample(min(n_samples, len(df_url)), random_state=42)
    df_html_sample = df_html.loc[df_url_sample.index]
    
    proc = StructuralProcessor({})
    # Get all base features except 'known_brand_mimicry' which we agreed to remove
    base_features = [f for f in proc.get_feature_names()[:26] if f != 'known_brand_mimicry']
    
    urls = df_url_sample['Data'].astype(str).values
    htmls = df_html_sample['Data'].astype(str).values
    
    data = []
    for u, h in zip(urls, htmls):
        domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
        feats = proc._extract_numerical_features(u, h, domain)
        # Filter out the specific feature index (10 is known_brand_mimicry)
        # It's safer to build a dict
        all_feats = proc.get_feature_names()[:26]
        row_dict = dict(zip(all_feats, feats))
        
        filtered_row = {k: v for k, v in row_dict.items() if k in base_features}
        data.append(filtered_row)
        
    return pd.DataFrame(data)

def generate_correlation_heatmap():
    print("Loading 2026 dataset for correlation analysis...")
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    print("Extracting features...")
    df_features = extract_features_for_corr(df_new_url, df_new_html, n_samples=3000)
    
    print("Computing Spearman correlation matrix...")
    # Spearman is better for non-linear structural features than Pearson
    corr_matrix = df_features.corr(method='spearman')
    
    # Create mask for the upper triangle to make it cleaner
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(20, 16))
    
    # Use a divergent colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=False) # Too many features for annotations, colors show it better
                
    plt.title("Structural Feature Orthogonality (Spearman Correlation Heatmap)", fontsize=24, fontweight='bold', pad=20)
    
    out_dir = r'd:\Desktop\PhishingDetection\docs\assets\explainable_ai'
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_correlation_orthogonality.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_dir}\\feature_correlation_orthogonality.png")

if __name__ == "__main__":
    generate_correlation_heatmap()

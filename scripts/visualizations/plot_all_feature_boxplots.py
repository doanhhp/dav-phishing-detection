import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Adjust the python path so we can import src
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.features.structural import StructuralProcessor
from urllib.parse import urlparse

def load_dataset(dataset_name, samples_per_class=500):
    print(f"Loading standardized {dataset_name} dataset...")
    
    file_map = {
        "Main": "main_dataset.parquet",
        "OOD": "ood_dataset.parquet",
        "PhreshPhish": "phreshphish_dataset.parquet"
    }
    
    file_path = project_root / 'data' / 'processed' / 'standardized' / file_map[dataset_name]
    df = pd.read_parquet(file_path)
    
    phish_df = df[df['Category'] == 1].sample(n=min(samples_per_class, len(df[df['Category'] == 1])), random_state=42)
    benign_df = df[df['Category'] == 0].sample(n=min(samples_per_class, len(df[df['Category'] == 0])), random_state=42)
    
    phish_df['Class'] = 'Phishing'
    benign_df['Class'] = 'Legitimate'
    return pd.concat([phish_df, benign_df], ignore_index=True)

def extract_features(df):
    print("Extracting 28 scalar base features...")
    proc = StructuralProcessor({})
    
    extracted_data = []
    
    for idx, row in df.iterrows():
        url = str(row['Data'])
        html = str(row['html']) if 'html' in row else ''
        domain = urlparse(url if url.startswith('http') else f"http://{url}").netloc
        
        # Extract numerical features directly using the processor's internal method
        num_features = proc._extract_numerical_features(url, html, domain)
        
        # Create a dict combining class and features
        feat_dict = {'Class': row['Class']}
        for i, feat_name in enumerate(proc.get_feature_names()[:28]):
            feat_dict[feat_name] = num_features[i]
            
        extracted_data.append(feat_dict)
        
    return pd.DataFrame(extracted_data)

def plot_distributions(df_features, dataset_name, output_dir):
    print(f"Plotting {dataset_name} distributions...")
    
    features = [c for c in df_features.columns if c != 'Class']
    
    # Create a 7x4 grid
    fig, axes = plt.subplots(7, 4, figsize=(24, 30))
    axes = axes.flatten()
    
    sns.set_theme(style="whitegrid")
    palette = {'Phishing': '#e74c3c', 'Legitimate': '#3498db'}
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Use boxplot for exact quantiles, no smoothing.
        # We disable fliers (outliers) to make the boxes actually readable, 
        # since HTML lengths and counts can have massive outliers compressing the box.
        sns.boxplot(data=df_features, x='Class', y=feature, ax=ax, palette=palette, showfliers=False)
        
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Value')
        
    # Hide any unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(f'Feature Distributions: {dataset_name} Dataset', fontsize=24, y=0.92, fontweight='bold')
    
    out_path = Path(output_dir) / f"{dataset_name}_feature_boxplots.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out_path}")

def main():
    output_dir = project_root / 'docs/assets/feature_distributions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in ["Main", "OOD", "PhreshPhish"]:
        df = load_dataset(dataset, samples_per_class=500)
        df_features = extract_features(df)
        plot_distributions(df_features, dataset, output_dir)
        
if __name__ == "__main__":
    main()

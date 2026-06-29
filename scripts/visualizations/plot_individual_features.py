import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.features.structural import StructuralProcessor
from urllib.parse import urlparse

def load_dataset(dataset_name, samples_per_class=1000):
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
    print("Extracting features...")
    proc = StructuralProcessor({})
    extracted_data = []
    for idx, row in df.iterrows():
        url = str(row['Data'])
        html = str(row['html']) if 'html' in row else ''
        domain = urlparse(url if url.startswith('http') else f"http://{url}").netloc
        num_features = proc._extract_numerical_features(url, html, domain)
        feat_dict = {'Class': row['Class']}
        for i, feat_name in enumerate(proc.get_feature_names()[:28]):
            feat_dict[feat_name] = num_features[i]
        extracted_data.append(feat_dict)
    return pd.DataFrame(extracted_data)

def plot_individual_features(df_features, dataset_name, output_dir):
    print(f"Plotting individual features for {dataset_name}...")
    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    palette = {'Phishing': '#e74c3c', 'Legitimate': '#3498db'}
    
    binary_features = ['is_https', 'url_has_login', 'url_hyphen_domain', 'foreign_form_action', 'brand_discrepancy']
    ratio_features = ['url_digit_ratio', 'html_text_ratio', 'html_external_link_ratio', 'html_empty_link_ratio', 'html_js_ratio', 'external_resource_ratio', 'html_input_to_p_ratio']
    discrete_features = ['iframe_count', 'css_hidden_count', 'html_password_input_count', 'url_num_subdomains', 'url_path_depth']
    
    features = [c for c in df_features.columns if c != 'Class']
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        if feature in binary_features:
            prop_df = df_features.groupby(['Class', feature]).size().reset_index(name='count')
            total_counts = df_features['Class'].value_counts()
            prop_df['Proportion'] = prop_df.apply(lambda r: r['count'] / total_counts[r['Class']], axis=1)
            prop_true = prop_df[prop_df[feature] == 1]
            if not prop_true.empty:
                sns.barplot(data=prop_true, x='Class', y='Proportion', palette=palette, order=['Legitimate', 'Phishing'])
                plt.title(f'Proportion of True ({feature})', fontsize=14, fontweight='bold')
                plt.ylim(0, max(prop_true['Proportion']) * 1.2)
            else:
                plt.title(f'{feature} (No True Values)', fontsize=14, fontweight='bold')
                
        elif feature in discrete_features:
            # Use a countplot/histogram for zero-inflated discrete counts
            # Limit the x-axis to the 99th percentile to avoid crazy long tails in the bar chart
            p99 = df_features[feature].quantile(0.99)
            filtered_df = df_features[df_features[feature] <= p99]
            
            sns.countplot(data=filtered_df, x=feature, hue='Class', palette=palette)
            plt.title(f'Distribution of {feature} (Counts)', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Websites')
            plt.xlabel(f'{feature} (values > 99th percentile hidden)')
            
        elif feature in ratio_features:
            sns.histplot(data=df_features, x=feature, hue='Class', bins=30, common_norm=False, stat='density', palette=palette, alpha=0.6, element='step')
            plt.title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
            
        else:
            # Boxenplot with Y-axis limited to the 99th percentile to fix the squished scale issue
            p99 = df_features[feature].quantile(0.99)
            
            # Filter out the extreme 1% outliers to make the plot completely readable
            filtered_df = df_features[df_features[feature] <= p99]
            
            sns.boxenplot(data=filtered_df, x='Class', y=feature, palette=palette, order=['Legitimate', 'Phishing'])
            plt.title(f'Distribution of {feature} (Top 1% Outliers Removed)', fontsize=14, fontweight='bold')
            
        out_path = dataset_dir / f"{feature}.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()

def main():
    output_dir = project_root / 'docs' / 'assets' / 'feature_distributions_individual'
    for dataset in ["Main", "OOD", "PhreshPhish"]:
        df = load_dataset(dataset, samples_per_class=1000)
        df_features = extract_features(df)
        plot_individual_features(df_features, dataset, output_dir)
        print(f"Finished {dataset}!")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def load_data(dataset_name):
    path = project_root / 'data' / 'processed' / 'standardized' / f'{dataset_name.lower()}_dataset.parquet'
    df = pd.read_parquet(path)
    return df

def main():
    print("Loading datasets...")
    df_main = load_data('Main')
    df_phresh = load_data('PhreshPhish')
    df_ood = load_data('OOD')
    
    # Filter for Legitimate (Category == 0)
    df_main_legit = df_main[df_main['Category'] == 0].copy()
    df_phresh_legit = df_phresh[df_phresh['Category'] == 0].copy()
    df_ood_legit = df_ood[df_ood['Category'] == 0].copy()
    
    # Extract url_length
    df_main_legit['url_length'] = df_main_legit['Data'].apply(lambda x: len(str(x)))
    df_phresh_legit['url_length'] = df_phresh_legit['Data'].apply(lambda x: len(str(x)))
    df_ood_legit['url_length'] = df_ood_legit['Data'].apply(lambda x: len(str(x)))
    
    df_main_legit['Dataset'] = '2021 Main Dataset\n(Historic)'
    df_phresh_legit['Dataset'] = '2024 PhreshPhish\n(Zero-Day)'
    df_ood_legit['Dataset'] = '2026 OOD\n(Crawler Bias)'
    
    # Combine
    combined = pd.concat([df_main_legit, df_phresh_legit, df_ood_legit])
    
    # Limit outliers for visualization
    p99 = combined['url_length'].quantile(0.95)
    filtered = combined[combined['url_length'] <= p99]
    
    out_dir = project_root / 'docs' / 'assets' / 'explainable_ai' / 'zero_day_failure'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot URL Length
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.violinplot(data=filtered, x='Dataset', y='url_length', palette=['#3498db', '#9b59b6', '#e74c3c'], inner="quartile")
    plt.title("Distribution of URL Length for Legitimate Websites Across Datasets", fontsize=14, fontweight='bold')
    plt.ylabel("URL Length (characters)")
    plt.xlabel("")
    
    plt.tight_layout()
    plt.savefig(out_dir / 'dataset_shift_url_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Let's also plot url_num_special_chars (non-alphanumeric)
    df_main_legit['special_chars'] = df_main_legit['Data'].apply(lambda x: sum(1 for c in str(x) if not c.isalnum()))
    df_phresh_legit['special_chars'] = df_phresh_legit['Data'].apply(lambda x: sum(1 for c in str(x) if not c.isalnum()))
    df_ood_legit['special_chars'] = df_ood_legit['Data'].apply(lambda x: sum(1 for c in str(x) if not c.isalnum()))
    
    combined_sp = pd.concat([df_main_legit, df_phresh_legit, df_ood_legit])
    p99_sp = combined_sp['special_chars'].quantile(0.95)
    filtered_sp = combined_sp[combined_sp['special_chars'] <= p99_sp]
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=filtered_sp, x='Dataset', y='special_chars', palette=['#3498db', '#9b59b6', '#e74c3c'], inner="quartile")
    plt.title("Distribution of Special Characters for Legitimate URLs Across Datasets", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Special Characters (e.g. /, -, ?)")
    plt.xlabel("")
    
    plt.tight_layout()
    plt.savefig(out_dir / 'dataset_shift_special_chars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dataset shift plots to {out_dir}")

if __name__ == "__main__":
    main()

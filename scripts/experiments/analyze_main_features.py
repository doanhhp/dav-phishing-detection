import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scripts.visualizations.plot_individual_features import load_dataset, extract_features

def main():
    print("Loading Main dataset...")
    df = load_dataset("Main", samples_per_class=2000)
    df_features = extract_features(df)
    
    binary_features = ['is_https', 'url_has_login', 'url_hyphen_domain', 'foreign_form_action', 'brand_discrepancy']
    
    print("\n--- FEATURE ANALYSIS (MAIN DATASET) ---")
    
    for feature in [c for c in df_features.columns if c != 'Class']:
        print(f"\nFeature: {feature}")
        if feature in binary_features:
            prop = df_features.groupby('Class')[feature].mean() * 100
            print(f"  Legitimate: {prop.get('Legitimate', 0):.2f}% True")
            print(f"  Phishing:   {prop.get('Phishing', 0):.2f}% True")
        else:
            stats = df_features.groupby('Class')[feature].agg(['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
            stats.columns = ['Median', 'Mean', '25th', '75th']
            print(f"  Legitimate - Median: {stats.loc['Legitimate', 'Median']:.2f} (IQR: {stats.loc['Legitimate', '25th']:.2f} - {stats.loc['Legitimate', '75th']:.2f})")
            print(f"  Phishing   - Median: {stats.loc['Phishing', 'Median']:.2f} (IQR: {stats.loc['Phishing', '25th']:.2f} - {stats.loc['Phishing', '75th']:.2f})")

if __name__ == "__main__":
    main()

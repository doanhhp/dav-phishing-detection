import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Datasets ---")
    df_old_url = pd.read_excel('data/raw/URL.xlsx')
    df_old_html = pd.read_excel('data/raw/html.xlsx')
    
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    old_spam_idx = df_old_url['Label'] == 'spam' if 'Label' in df_old_url.columns else df_old_url['Category'] == 'spam'
    df_old_phish_url = df_old_url[old_spam_idx].head(2000)
    df_old_phish_html = df_old_html[old_spam_idx].head(2000)
    
    new_spam_idx = df_new_url['Category'] == 'spam'
    df_new_phish_url = df_new_url[new_spam_idx].head(2000)
    df_new_phish_html = df_new_html[new_spam_idx].head(2000)
    
    print("--- 2. Extracting Structural Features ---")
    proc = StructuralProcessor({})
    
    df_old = df_old_phish_url[['Data']].copy()
    df_old['html'] = df_old_phish_html['Data'].values
    X_old = proc.fit_transform(df_old)
    
    df_new = df_new_phish_url[['Data']].copy()
    df_new['html'] = df_new_phish_html['Data'].values
    X_new = proc.transform(df_new)
    
    feature_names = proc.get_feature_names()
    
    # We will compute the Kolmogorov-Smirnov statistic for each feature
    # KS-stat is between 0.0 (identical distributions) and 1.0 (completely disjoint)
    results = []
    
    for i, feature in enumerate(feature_names):
        old_vals = X_old[:, i]
        new_vals = X_new[:, i]
        
        # KS Test
        ks_stat, p_value = ks_2samp(old_vals, new_vals)
        
        # Mean shift
        old_mean = np.mean(old_vals)
        new_mean = np.mean(new_vals)
        shift_pct = 0
        if old_mean != 0:
            shift_pct = ((new_mean - old_mean) / abs(old_mean)) * 100
            
        results.append({
            'Feature': feature,
            'Drift Score (KS)': ks_stat,
            '2021 Mean': old_mean,
            '2026 Mean': new_mean,
            'Shift %': shift_pct
        })
        
    df_results = pd.DataFrame(results)
    
    # Sort by Drift Score (Ascending = Most Stable first)
    df_results = df_results.sort_values(by='Drift Score (KS)', ascending=True)
    
    # Generate Markdown Report
    report = "# Numerical Feature Drift Analysis (2021 vs 2026)\n\n"
    report += "This report mathematically ranks how much each feature mutated over 5 years. "
    report += "We use the **Kolmogorov-Smirnov (KS) Statistic**, which measures the maximum difference between the two distributions.\n\n"
    report += "*   **Drift Score near 0.0:** The feature is completely stable (a structural invariant).\n"
    report += "*   **Drift Score near 1.0:** The feature has mutated completely (high drift).\n\n"
    
    # Separate into Top 10 Most Stable and Top 10 Most Drifted
    report += "### Top 15 Most Stable Features (The Core Anchors)\n\n"
    report += "| Feature | Drift Score (0-1) | 2021 Mean | 2026 Mean | Shift % |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in df_results.head(15).iterrows():
        report += f"| `{row['Feature']}` | **{row['Drift Score (KS)']:.3f}** | {row['2021 Mean']:.3f} | {row['2026 Mean']:.3f} | {row['Shift %']:.1f}% |\n"
        
    report += "\n### Top 15 Most Mutated Features (High Concept Drift)\n\n"
    report += "| Feature | Drift Score (0-1) | 2021 Mean | 2026 Mean | Shift % |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in df_results.tail(15).iloc[::-1].iterrows():
        report += f"| `{row['Feature']}` | **{row['Drift Score (KS)']:.3f}** | {row['2021 Mean']:.3f} | {row['2026 Mean']:.3f} | {row['Shift %']:.1f}% |\n"
        
    # Save to artifact
    output_path = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_drift_report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Saved numerical drift report to: {output_path}")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Datasets ---")
    df_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    if 'Label' in df_url.columns:
        df_url = df_url.rename(columns={'Label': 'Category'})
        
    print("--- 2. Extracting Structural Features ---")
    proc = StructuralProcessor({})
    df_raw = df_url[['Data']].copy()
    df_raw['html'] = df_html['Data'].values
    
    X = proc.fit_transform(df_raw)
    feature_names = proc.get_feature_names()
    
    df_feat = pd.DataFrame(X, columns=feature_names)
    df_feat['Class'] = df_url['Category'].map({'ham': 0, 'spam': 1})
    
    print("--- 3. Generating Correlation Heatmap (Multicollinearity) ---")
    # Select top 15 features with highest variance to keep heatmap readable
    variances = df_feat.drop('Class', axis=1).var().sort_values(ascending=False)
    top_15_features = variances.head(15).index.tolist()
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df_feat[top_15_features + ['Class']].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Feature Correlation Heatmap (Redundancy Analysis)", fontsize=16, fontweight='bold')
    heatmap_path = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Heatmap to: {heatmap_path}")
    
    print("--- 4. Generating Radar Chart (Profile Comparison) ---")
    # Group features into conceptual categories to show the "footprint" of an attack
    categories = ['DOM Complexity', 'Lexical Suspicion', 'CSS Evasion', 'External Links', 'Form Actions']
    
    # Map features to categories (safely checking if they exist)
    cat_map = {
        'DOM Complexity': ['dom_depth', 'html_num_tags', 'tag_diversity'],
        'Lexical Suspicion': ['url_length', 'url_entropy', 'url_num_special_chars'],
        'CSS Evasion': ['css_hidden_count'],
        'External Links': ['html_external_link_ratio'],
        'Form Actions': ['foreign_form_action', 'html_password_input_count']
    }
    
    # Calculate mean normalized score for each category
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_feat.drop('Class', axis=1)), columns=feature_names)
    df_normalized['Class'] = df_feat['Class']
    
    ham_means = []
    spam_means = []
    
    for cat in categories:
        valid_features = [f for f in cat_map[cat] if f in feature_names]
        if valid_features:
            ham_means.append(df_normalized[df_normalized['Class'] == 0][valid_features].mean().mean())
            spam_means.append(df_normalized[df_normalized['Class'] == 1][valid_features].mean().mean())
        else:
            ham_means.append(0)
            spam_means.append(0)
            
    # Radar chart plotting
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ham_means += ham_means[:1]
    spam_means += spam_means[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, ham_means, color='green', linewidth=2, label='Legitimate (Ham)')
    ax.fill(angles, ham_means, color='green', alpha=0.25)
    ax.plot(angles, spam_means, color='red', linewidth=2, label='Phishing (Spam)')
    ax.fill(angles, spam_means, color='red', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    plt.title("Phishing vs Legitimate Profile (Radar Chart)", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    radar_path = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_radar_profile.png'
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Radar Chart to: {radar_path}")
    
    print("--- 5. Generating Boxplots (Outlier & Quartile Analysis) ---")
    features_to_box = ['url_length', 'dom_depth', 'html_text_ratio', 'tag_diversity']
    features_to_box = [f for f in features_to_box if f in feature_names]
    
    fig, axes = plt.subplots(1, len(features_to_box), figsize=(18, 6))
    df_feat['Class Name'] = df_feat['Class'].map({0: 'Legitimate', 1: 'Phishing'})
    
    for i, feature in enumerate(features_to_box):
        sns.boxplot(x='Class Name', y=feature, data=df_feat, ax=axes[i], palette={'Legitimate': 'lightgreen', 'Phishing': 'lightcoral'})
        axes[i].set_title(f'Outlier Analysis: {feature}', fontweight='bold')
        axes[i].set_xlabel('')
        
    plt.tight_layout()
    boxplot_path = r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\feature_outlier_boxplots.png'
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Boxplots to: {boxplot_path}")

if __name__ == "__main__":
    main()

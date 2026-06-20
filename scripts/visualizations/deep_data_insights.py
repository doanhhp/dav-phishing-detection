import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Datasets ---")
    df_old_url = pd.read_excel('data/raw/URL.xlsx')
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    if 'Label' in df_old_url.columns:
        df_old_url = df_old_url.rename(columns={'Label': 'Category'})
    if 'Label' in df_new_url.columns:
        df_new_url = df_new_url.rename(columns={'Label': 'Category'})
        
    # Extract structural features for the scatter/violin plots
    proc = StructuralProcessor({})
    df_raw = df_new_url[['Data']].copy()
    df_raw['html'] = df_new_html['Data'].values
    X = proc.fit_transform(df_raw)
    feature_names = proc.get_feature_names()
    df_feat = pd.DataFrame(X, columns=feature_names)
    df_feat['Class'] = df_new_url['Category'].map({'ham': 'Legitimate', 'spam': 'Phishing'})

    # ---------------------------------------------------------
    # INSIGHT 1: URL Keywords (What words do scammers use?)
    # ---------------------------------------------------------
    print("Generating Insight 1: URL Keywords...")
    phish_urls = df_new_url[df_new_url['Category'] == 'spam']['Data'].astype(str)
    
    words = []
    for url in phish_urls:
        # Extract words from URL path and domain
        parsed = urlparse(url if '://' in url else 'http://' + url)
        text = parsed.netloc + parsed.path
        # Split by non-alphanumeric
        tokens = re.split(r'[^a-zA-Z]', text.lower())
        words.extend([t for t in tokens if len(t) > 3 and t not in ['http', 'https', 'www', 'com', 'net', 'org', 'html', 'php']])
        
    top_words = Counter(words).most_common(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[w[1] for w in top_words], y=[w[0] for w in top_words], palette='Reds_r')
    plt.title("Insight 1: Top Keywords Hidden in 2026 Phishing URLs", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\insight_1_url_keywords.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # INSIGHT 2: The "Hollow Shell" Effect (Violin Plots)
    # ---------------------------------------------------------
    print("Generating Insight 2: Hollow Shell Violin Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.violinplot(data=df_feat, x='Class', y='dom_depth', ax=ax1, palette={'Legitimate': 'lightgreen', 'Phishing': 'lightcoral'})
    ax1.set_title("Insight 2A: Maximum DOM Depth\n(Legit sites are complex; Phishing sites are shallow shells)", fontsize=12, fontweight='bold')
    
    # Use log scale for length to make violin readable
    df_feat['log_html_length'] = np.log1p(df_feat['html_length'])
    sns.violinplot(data=df_feat, x='Class', y='log_html_length', ax=ax2, palette={'Legitimate': 'lightgreen', 'Phishing': 'lightcoral'})
    ax2.set_title("Insight 2B: HTML Source Code Length (Log Scale)\n(Phishing sites have vastly less code)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\insight_2_hollow_shell.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # INSIGHT 3: URL Length vs Entropy (Scatter Plot)
    # ---------------------------------------------------------
    print("Generating Insight 3: URL Length vs Entropy Scatter...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_feat, x='url_length', y='url_entropy', hue='Class', 
                    palette={'Legitimate': 'green', 'Phishing': 'red'}, alpha=0.6, s=50)
    plt.title("Insight 3: The 'Randomized Token' Attack Vector\n(Phishers use extremely long, highly entropic URLs)", fontsize=14, fontweight='bold')
    plt.xlabel("URL Length")
    plt.ylabel("URL Entropy (Randomness)")
    plt.tight_layout()
    plt.savefig(r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\insight_3_url_scatter.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # INSIGHT 4: The HTTPS Evolution (Pie Charts)
    # ---------------------------------------------------------
    print("Generating Insight 4: HTTPS Evolution Pie Charts...")
    
    # 2021 Phishing HTTPS ratio
    old_phish = df_old_url[df_old_url['Category'] == 'spam']['Data'].astype(str)
    old_https = old_phish.str.startswith('https').mean() * 100
    
    # 2026 Phishing HTTPS ratio
    new_phish = df_new_url[df_new_url['Category'] == 'spam']['Data'].astype(str)
    new_https = new_phish.str.startswith('https').mean() * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.pie([old_https, 100-old_https], labels=['HTTPS (Secure)', 'HTTP (Insecure)'], 
            autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90)
    ax1.set_title("Insight 4A: 2021 Phishing HTTPS Usage\n(Scammers didn't bother with SSL)", fontsize=12, fontweight='bold')
    
    ax2.pie([new_https, 100-new_https], labels=['HTTPS (Secure)', 'HTTP (Insecure)'], 
            autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90)
    ax2.set_title("Insight 4B: 2026 Phishing HTTPS Usage\n(Scammers adapted to bypass browser warnings)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\insight_4_https_evolution.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # INSIGHT 5: Top HTML Tags (Grouped Bar Chart)
    # ---------------------------------------------------------
    print("Generating Insight 5: Top HTML Tags...")
    # Sample 500 of each to parse quickly
    ham_html = df_new_html[df_new_url['Category'] == 'ham']['Data'].astype(str).head(500)
    spam_html = df_new_html[df_new_url['Category'] == 'spam']['Data'].astype(str).head(500)
    
    def count_tags(html_series):
        tag_counts = Counter()
        for html in html_series:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                tags = [tag.name for tag in soup.find_all()]
                tag_counts.update(tags)
            except: pass
        # Average per page
        return {k: v / len(html_series) for k, v in tag_counts.items()}
        
    ham_tags = count_tags(ham_html)
    spam_tags = count_tags(spam_html)
    
    target_tags = ['div', 'span', 'a', 'script', 'form', 'input', 'iframe']
    
    df_tags = pd.DataFrame({
        'Tag': target_tags * 2,
        'Average Count Per Page': [ham_tags.get(t, 0) for t in target_tags] + [spam_tags.get(t, 0) for t in target_tags],
        'Class': ['Legitimate']*len(target_tags) + ['Phishing']*len(target_tags)
    })
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_tags, x='Tag', y='Average Count Per Page', hue='Class', palette={'Legitimate': 'green', 'Phishing': 'red'})
    plt.title("Insight 5: Average HTML Tags per Page\n(Legit sites style heavily with Div/Span; Phish sites focus on Forms/Inputs)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Doanh1\.gemini\antigravity\brain\4a4ead50-6758-44bf-8d9b-6d0f705fe101\insight_5_html_tags.png', dpi=300)
    plt.close()

    print("All deep insight visualizations generated successfully.")

if __name__ == "__main__":
    main()

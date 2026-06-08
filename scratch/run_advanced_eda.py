import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def get_advanced_html_features(html_content):
    f = {'dead_link_ratio': 0, 'text_to_code_ratio': 0, 'input_density': 0}
    if not isinstance(html_content, str) or not html_content:
        return f
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Dead Link Ratio
    links = soup.find_all('a')
    if links:
        dead_links = [a for a in links if a.get('href') in ['#', 'javascript:void(0)', '', None]]
        f['dead_link_ratio'] = len(dead_links) / len(links)
        
    # 2. Text-to-Code Ratio
    text = soup.get_text(separator=' ', strip=True)
    text_len = len(text)
    code_len = len(html_content)
    if code_len > 0:
        f['text_to_code_ratio'] = text_len / code_len
        
    # 3. Input Density
    all_tags = soup.find_all(True)
    inputs = soup.find_all('input')
    if all_tags:
        f['input_density'] = (len(inputs) / len(all_tags)) * 100 # per 100 tags
        
    return f

def get_advanced_url_features(url):
    f = {'url_path_depth': 0, 'subdomain_count': 0}
    if not isinstance(url, str) or not url:
        return f
        
    url_str = url if url.startswith('http') else 'http://' + url
    parsed = urlparse(url_str)
    
    # 1. Path Depth
    path = parsed.path
    if path:
        f['url_path_depth'] = path.count('/')
        
    # 2. Subdomain Count
    netloc = parsed.netloc
    if netloc:
        parts = netloc.split('.')
        f['subdomain_count'] = len(parts) - 2 if len(parts) > 2 else 0
        
    return f

def main():
    print("--- 1. Loading Datasets ---")
    
    # Load Old Samples (Sampled for speed)
    try:
        df_old_url = pd.read_excel(project_root / 'data/raw/URL.xlsx', engine='calamine').sample(1500, random_state=42)
        df_old_html = pd.read_excel(project_root / 'data/raw/html.xlsx', engine='calamine')
    except:
        df_old_url = pd.read_excel(project_root / 'data/raw/URL.xlsx').sample(1500, random_state=42)
        df_old_html = pd.read_excel(project_root / 'data/raw/html.xlsx')
        
    df_old = df_old_url.copy()
    df_old['html'] = df_old_html.loc[df_old.index, 'Data']
    df_old['url'] = df_old['Data']
    df_old['Dataset'] = 'Historical (2021)'
    
    # Load all OOD Samples
    try:
        df_ood_url = pd.read_excel(project_root / 'data/raw/OOD_URL.xlsx', engine='calamine')
        df_ood_html = pd.read_excel(project_root / 'data/raw/OOD_html.xlsx', engine='calamine')
    except:
        df_ood_url = pd.read_excel(project_root / 'data/raw/OOD_URL.xlsx')
        df_ood_html = pd.read_excel(project_root / 'data/raw/OOD_html.xlsx')

    df_ood = df_ood_url.copy()
    df_ood['html'] = df_ood_html['Data']
    df_ood['url'] = df_ood['Data']
    df_ood['Dataset'] = 'Zero-Day (2026)'

    df = pd.concat([df_old, df_ood], ignore_index=True)
    df['label_name'] = df['Category'].map({'ham': 'Legitimate', 'spam': 'Phishing'})

    print(f"Total samples for Advanced EDA: {len(df)}")

    print("\n--- 2. Extracting Advanced Features ---")
    features = []
    for idx, row in df.iterrows():
        f_html = get_advanced_html_features(row['html'])
        f_url = get_advanced_url_features(row['url'])
        f_html.update(f_url)
        features.append(f_html)

    df_feat = pd.DataFrame(features)
    df_final = pd.concat([df.reset_index(drop=True), df_feat], axis=1)

    print("\n--- 3. Generating Visualizations ---")
    docs_dir = project_root / 'docs'
    assets_dir = docs_dir / 'assets' / 'zero_day_analysis'
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Text-to-Code Ratio
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_final, x='label_name', y='text_to_code_ratio', hue='Dataset', palette='Set2')
    plt.title('Text-to-Code Ratio (Content vs HTML)')
    plt.ylabel('Ratio (Visible Text / Total HTML Length)')
    plt.ylim(0, 0.5) # Zoom in to ignore huge text outliers
    plt.tight_layout()
    plt.savefig(assets_dir / 'adv_eda_text_to_code.png', dpi=300)

    # Plot 2: Dead Link Ratio
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_final, x='label_name', y='dead_link_ratio', hue='Dataset', palette='Set1', estimator=np.mean)
    plt.title('Dead Link Ratio (The "Laziness" Indicator)')
    plt.ylabel('Percentage of <a> tags linking to "#" or null')
    plt.tight_layout()
    plt.savefig(assets_dir / 'adv_eda_dead_links.png', dpi=300)

    # Plot 3: Input Density
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_final, x='label_name', y='input_density', hue='Dataset', palette='Set2')
    plt.title('Input Tag Density (Hyper-Focus on Stealing Data)')
    plt.ylabel('Number of <input> tags per 100 HTML tags')
    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(assets_dir / 'adv_eda_input_density.png', dpi=300)

    # Plot 4: URL Subdomains and Depth
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=df_final, x='label_name', y='subdomain_count', hue='Dataset', ax=axes[0], palette='muted', estimator=np.mean)
    axes[0].set_title('Average Subdomain Count')
    axes[0].set_ylabel('Number of Subdomains (e.g. login.update.paypal.com)')
    
    sns.barplot(data=df_final, x='label_name', y='url_path_depth', hue='Dataset', ax=axes[1], palette='muted', estimator=np.mean)
    axes[1].set_title('Average URL Path Depth')
    axes[1].set_ylabel('Number of Directories (e.g. /secure/auth/step1)')
    plt.tight_layout()
    plt.savefig(assets_dir / 'adv_eda_url_obfuscation.png', dpi=300)

    print("\n--- 4. Updating Reports ---")
    
    # Update exploratory_data_analysis.md
    report_path = docs_dir / 'reports' / 'exploratory_data_analysis.md'
    if report_path.exists():
        with open(report_path, 'a') as f:
            f.write("\n\n---\n# Part: Advanced Behavioral Properties\n---\n\n")
            f.write("To understand *why* the structures differ, we extracted behavioral quirks that expose the malicious intent of scammers.\n\n")
            f.write("### 1. The 'Dead Link' Phenomenon\n")
            f.write("Legitimate sites have rich navigation. Phishers only care about the login form and leave other links empty (`href=\"#\"`).\n")
            f.write("![Dead Links](../assets/zero_day_analysis/adv_eda_dead_links.png)\n\n")
            
            f.write("### 2. The Text-to-Code Ratio\n")
            f.write("Phishing sites are often just a background image and an input box, meaning they have almost no actual readable text compared to the massive amount of HTML code.\n")
            f.write("![Text to Code](../assets/zero_day_analysis/adv_eda_text_to_code.png)\n\n")
            
            f.write("### 3. Hyper-Focus on Inputs\n")
            f.write("Phishing sites have an unnaturally high density of `<input>` fields relative to the rest of the page.\n")
            f.write("![Input Density](../assets/zero_day_analysis/adv_eda_input_density.png)\n\n")
            
            f.write("### 4. Deep Obfuscation in URLs\n")
            f.write("Scammers hide the real domain by using many subdomains or deep paths.\n")
            f.write("![URL Obfuscation](../assets/zero_day_analysis/adv_eda_url_obfuscation.png)\n\n")

    print("Advanced EDA Complete! Graphs saved and reports updated.")

if __name__ == "__main__":
    main()

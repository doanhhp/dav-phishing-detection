import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import math
from urllib.parse import urlparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def get_dom_depth(soup):
    if not soup or not soup.html: return 0
    max_depth = 0
    for tag in soup.find_all(True):
        depth = len(list(tag.parents))
        if depth > max_depth:
            max_depth = depth
    return max_depth

def get_tag_diversity(soup):
    if not soup: return 0
    tags = [tag.name for tag in soup.find_all(True)]
    return len(set(tags))

def get_external_resource_ratio(soup, url):
    if not soup: return 0
    domain = urlparse(url if str(url).startswith('http') else 'http://'+str(url)).netloc
    
    resources = soup.find_all(['img', 'script', 'link', 'iframe'])
    if not resources: return 0
    
    external = 0
    for tag in resources:
        src = tag.get('src') or tag.get('href')
        if src and str(src).startswith('http') and domain not in str(src):
            external += 1
            
    return external / len(resources)

def calculate_entropy(string):
    if not isinstance(string, str) or not string: return 0
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    return - sum([p * math.log(p) / math.log(2.0) for p in prob])

def count_hidden_elements(soup):
    if not soup: return 0
    hidden_tags = soup.find_all(style=lambda value: value and ('display:none' in value.replace(' ', '') or 'visibility:hidden' in value.replace(' ', '')))
    return len(hidden_tags)

def main():
    print("--- 1. Loading OOD Data ---")
    try:
        df_ood_url = pd.read_excel(project_root / 'data/raw/OOD_URL.xlsx', engine='calamine')
        df_ood_html = pd.read_excel(project_root / 'data/raw/OOD_html.xlsx', engine='calamine')
    except Exception as e:
        print(f"Calamine failed: {e}. Falling back to default.")
        df_ood_url = pd.read_excel(project_root / 'data/raw/OOD_URL.xlsx')
        df_ood_html = pd.read_excel(project_root / 'data/raw/OOD_html.xlsx')

    df = df_ood_url.copy()
    df['html'] = df_ood_html['Data']
    df['url'] = df['Data']
    df['label_name'] = df['Category'].map({'ham': 'Legitimate', 'spam': 'Phishing'})

    print(f"Total OOD samples: {len(df)}")
    print(df['label_name'].value_counts())

    print("\n--- 2. Extracting Features ---")
    features = []
    for idx, row in df.iterrows():
        try:
            soup = BeautifulSoup(str(row['html']), 'html.parser')
            url_str = str(row['url'])
            f = {
                'url_length': len(url_str),
                'url_entropy': calculate_entropy(url_str),
                'html_length': len(str(row['html'])),
                'dom_depth': get_dom_depth(soup),
                'tag_diversity': get_tag_diversity(soup),
                'external_resource_ratio': get_external_resource_ratio(soup, url_str),
                'hidden_elements': count_hidden_elements(soup),
                'num_iframes': len(soup.find_all('iframe')),
                'num_scripts': len(soup.find_all('script'))
            }
        except Exception as e:
            f = {'url_length': 0, 'url_entropy': 0, 'html_length': 0, 'dom_depth': 0, 'tag_diversity': 0, 'external_resource_ratio': 0, 'hidden_elements': 0, 'num_iframes': 0, 'num_scripts': 0}
        features.append(f)

    df_feat = pd.DataFrame(features)
    df_final = pd.concat([df.reset_index(drop=True), df_feat], axis=1)

    print("\n--- 3. Generating Visualizations ---")
    docs_dir = project_root / 'docs'
    assets_dir = docs_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: URL Characteristics
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.kdeplot(data=df_final, x='url_length', hue='label_name', ax=axes[0], fill=True, common_norm=False)
    axes[0].set_title('URL Length Distribution (Zero-Day)')
    axes[0].set_xlim(0, 150)
    
    sns.kdeplot(data=df_final, x='url_entropy', hue='label_name', ax=axes[1], fill=True, common_norm=False)
    axes[1].set_title('URL Entropy Distribution (Zero-Day)')
    plt.tight_layout()
    plt.savefig(assets_dir / 'ood_eda_url.png', dpi=300)

    # Plot 2: HTML Structural Complexity
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=df_final, x='label_name', y='dom_depth', ax=axes[0], palette='Set2')
    axes[0].set_title('DOM Tree Depth: Phishing vs Legitimate')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Max DOM Depth (Log Scale)')
    
    sns.boxplot(data=df_final, x='label_name', y='tag_diversity', ax=axes[1], palette='Set2')
    axes[1].set_title('HTML Tag Diversity: Phishing vs Legitimate')
    axes[1].set_ylabel('Number of Unique HTML Tags')
    plt.tight_layout()
    plt.savefig(assets_dir / 'ood_eda_structure.png', dpi=300)

    # Plot 3: Evasion & Malicious Indicators
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.barplot(data=df_final, x='label_name', y='hidden_elements', ax=axes[0], palette='Set1', estimator=np.mean)
    axes[0].set_title('Avg Hidden Elements (display:none)')
    
    sns.barplot(data=df_final, x='label_name', y='num_iframes', ax=axes[1], palette='Set1', estimator=np.mean)
    axes[1].set_title('Avg Iframes per Page')
    
    sns.violinplot(data=df_final, x='label_name', y='external_resource_ratio', ax=axes[2], palette='muted')
    axes[2].set_title('External Resource Ratio')
    
    plt.tight_layout()
    plt.savefig(assets_dir / 'ood_eda_evasion.png', dpi=300)

    print("\n--- 4. Saving Statistical Report ---")
    stats_df = df_final.groupby('label_name')[['url_length', 'url_entropy', 'html_length', 'dom_depth', 'tag_diversity', 'hidden_elements', 'num_iframes', 'external_resource_ratio']].mean().round(3)
    
    report_path = docs_dir / 'ood_eda_report.md'
    with open(report_path, 'w') as f:
        f.write("# Zero-Day (OOD) Exploratory Data Analysis\n\n")
        f.write("This report analyzes the structural and textual characteristics of the 2026 Zero-Day (OOD) dataset to understand the modern threat landscape.\n\n")
        f.write("## 1. Summary Statistics (Averages)\n\n")
        f.write(stats_df.to_markdown())
        f.write("\n\n")
        f.write("## 2. Analytical Findings\n\n")
        f.write("### URL Patterns\n")
        f.write("![URL Distribution](assets/zero_day_analysis/ood_eda_url.png)\n\n")
        f.write("### Structural Complexity (The Simplicity Paradox)\n")
        f.write("![Structural Complexity](assets/zero_day_analysis/ood_eda_structure.png)\n\n")
        f.write("### Evasion Tactics (Iframes & Hidden Elements)\n")
        f.write("![Evasion Tactics](assets/zero_day_analysis/ood_eda_evasion.png)\n\n")
        
    print(f"Done! Report saved to {report_path}")

if __name__ == "__main__":
    main()

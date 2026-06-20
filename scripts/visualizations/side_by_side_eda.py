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
    print("--- 1. Loading Datasets ---")
    
    # Load 1500 Old Samples
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

    print(f"Total samples for side-by-side EDA: {len(df)}")

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

    print("\n--- 3. Generating Side-By-Side Visualizations ---")
    docs_dir = project_root / 'docs'
    assets_dir = docs_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: URL Characteristics
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.kdeplot(data=df_final[df_final['Category']=='spam'], x='url_length', hue='Dataset', ax=axes[0], fill=True, common_norm=False)
    axes[0].set_title('Phishing URL Length Comparison')
    axes[0].set_xlim(0, 150)
    
    sns.kdeplot(data=df_final[df_final['Category']=='spam'], x='url_entropy', hue='Dataset', ax=axes[1], fill=True, common_norm=False)
    axes[1].set_title('Phishing URL Entropy Comparison')
    plt.tight_layout()
    plt.savefig(assets_dir / 'side_by_side_url.png', dpi=300)

    # Plot 2: HTML Structural Complexity
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=df_final, x='label_name', y='dom_depth', hue='Dataset', ax=axes[0], palette='Set2')
    axes[0].set_title('DOM Tree Depth: 2021 vs 2026')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Max DOM Depth (Log Scale)')
    
    sns.boxplot(data=df_final, x='label_name', y='tag_diversity', hue='Dataset', ax=axes[1], palette='Set2')
    axes[1].set_title('HTML Tag Diversity: 2021 vs 2026')
    axes[1].set_ylabel('Number of Unique HTML Tags')
    plt.tight_layout()
    plt.savefig(assets_dir / 'side_by_side_structure.png', dpi=300)

    # Plot 3: Evasion & Malicious Indicators
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=df_final, x='label_name', y='hidden_elements', hue='Dataset', ax=axes[0], palette='Set1', estimator=np.mean)
    axes[0].set_title('Avg Hidden Elements (display:none)')
    
    sns.barplot(data=df_final, x='label_name', y='num_iframes', hue='Dataset', ax=axes[1], palette='Set1', estimator=np.mean)
    axes[1].set_title('Avg Iframes per Page')
    
    plt.tight_layout()
    plt.savefig(assets_dir / 'side_by_side_evasion.png', dpi=300)

    print("\n--- 4. Saving Summary ---")
    stats_df = df_final.groupby(['Dataset', 'label_name'])[['dom_depth', 'tag_diversity', 'hidden_elements', 'num_iframes']].mean().round(3)
    
    report_path = docs_dir / 'side_by_side_eda_report.md'
    with open(report_path, 'w') as f:
        f.write("# Side-By-Side EDA: 2021 vs 2026\n\n")
        f.write("This report compares the structural characteristics of the historical 2021 dataset against the modern 2026 zero-day dataset to map the evolution of phishing.\n\n")
        f.write("## 1. Summary Statistics (Averages)\n\n")
        f.write(stats_df.to_markdown())
        f.write("\n\n")
        f.write("## 2. Visualizations\n\n")
        f.write("### The Evolution of Phishing URLs\n")
        f.write("![URL Distribution](assets/domain_shift/side_by_side_url.png)\n\n")
        f.write("### Structural Shift\n")
        f.write("![Structural Complexity](assets/domain_shift/side_by_side_structure.png)\n\n")
        f.write("### Modern Evasion Tactics\n")
        f.write("![Evasion Tactics](assets/domain_shift/side_by_side_evasion.png)\n\n")
        
    print(f"Done! Report saved to {report_path}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.features.structural import StructuralProcessor

def remove_outliers(series, p=0.99):
    """Replace values above the p-th percentile with NaN to ignore them in plot"""
    threshold = series.quantile(p)
    return series.apply(lambda x: x if pd.isna(x) or x <= threshold else np.nan)

# Load dataset and sample
df = pd.read_parquet('data/processed/standardized/main_dataset.parquet')
if 'Category' in df.columns:
    df = df.rename(columns={'Category': 'label'})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'Phishing' or x == 1 else 0)

# We use a larger sample to get a good distribution curve
df_phish = df[df['label'] == 1].sample(n=10000, random_state=42)
df_benign = df[df['label'] == 0].sample(n=10000, random_state=42)
df_sampled = pd.concat([df_phish, df_benign]).copy()

processor = StructuralProcessor({})
base_features_names = processor.get_feature_names()[:28]

def extract_raw_features(row):
    url = str(row['Data']).lower()
    html = str(row['html']).lower()
    parsed = urlparse(url if url.startswith('http') else f"http://{url}")
    domain = parsed.netloc
    raw_vals = processor._extract_numerical_features(url, html, domain)
    return pd.Series(dict(zip(base_features_names, raw_vals)))

print("Extracting features...", file=sys.stderr)
features_df = df_sampled.apply(extract_raw_features, axis=1)
features_df['Label'] = df_sampled['label'].map({0: 'Benign', 1: 'Phishing'})

# --- Plot URL Features ---
url_feats_to_plot = ['url_length', 'url_num_dots', 'url_num_special_chars', 'url_path_depth']
plot_df_url = features_df[['Label'] + url_feats_to_plot].copy()

# Remove 99th percentile outliers per feature
for feat in url_feats_to_plot:
    plot_df_url[feat] = remove_outliers(plot_df_url[feat])

plot_df_url = plot_df_url.melt(id_vars=['Label'], value_vars=url_feats_to_plot, var_name='Feature', value_name='Value')
plot_df_url = plot_df_url.dropna()

plt.figure(figsize=(12, 6))
sns.stripplot(data=plot_df_url, x='Feature', y='Value', hue='Label', 
              dodge=True, alpha=0.3, jitter=True, size=2, palette=['#1f77b4', '#ff7f0e'])
plt.title("Distribution of URL characteristics with 99th percentile outliers removed")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.legend(title="Class")
plt.tight_layout()
Path('docs/assets/dataset_stats').mkdir(parents=True, exist_ok=True)
plt.savefig('docs/assets/dataset_stats/url_dist.png', dpi=300)
plt.close()

# --- Plot DOM Features ---
dom_feats_to_plot = ['html_num_tags', 'body_tag_count', 'html_script_count', 'tag_diversity']
plot_df_dom = features_df[['Label'] + dom_feats_to_plot].copy()

# Remove 99th percentile outliers per feature
for feat in dom_feats_to_plot:
    plot_df_dom[feat] = remove_outliers(plot_df_dom[feat])

plot_df_dom = plot_df_dom.melt(id_vars=['Label'], value_vars=dom_feats_to_plot, var_name='Feature', value_name='Value')
plot_df_dom = plot_df_dom.dropna()

plt.figure(figsize=(12, 6))
sns.stripplot(data=plot_df_dom, x='Feature', y='Value', hue='Label', 
              dodge=True, alpha=0.3, jitter=True, size=2, palette=['#1f77b4', '#ff7f0e'])
plt.title("Distribution of HTML characteristics with 99th percentile outliers removed")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.legend(title="Class")
plt.tight_layout()
plt.savefig('docs/assets/dataset_stats/dom_dist.png', dpi=300)
plt.close()

print("Plots saved to docs/assets/dataset_stats/", file=sys.stderr)

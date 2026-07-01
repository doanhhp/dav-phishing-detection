import pandas as pd
import numpy as np
import sys
from pathlib import Path
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.structural import StructuralProcessor

# Load dataset and sample
df = pd.read_parquet('data/processed/standardized/main_dataset.parquet')
if 'Category' in df.columns:
    df = df.rename(columns={'Category': 'label'})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'Phishing' or x == 1 else 0)

# Sample to speed up extraction
df_phish = df[df['label'] == 1].sample(n=5000, random_state=42)
df_benign = df[df['label'] == 0].sample(n=5000, random_state=42)
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
features_df['label'] = df_sampled['label'].values

url_features = [
    ('url_length', 'Total length of the URL (chars)'),
    ('url_num_dots', 'Number of dots (.) in URL'),
    ('url_num_special_chars', 'Number of special chars in URL'),
    ('url_digit_ratio', 'Ratio of digits to total characters'),
    ('url_num_subdomains', 'Number of subdomains'),
    ('url_entropy', 'Shannon entropy of URL string'),
    ('url_path_depth', 'Number of directories in path')
]

dom_features = [
    ('html_length', 'Total length of HTML source (chars)'),
    ('html_num_tags', 'Total number of HTML tags'),
    ('html_script_count', 'Number of \\textless{}script\\textgreater{} tags'),
    ('html_title_length', 'Length of page title'),
    ('dom_depth', 'Maximum depth of the DOM tree'),
    ('tag_diversity', 'Number of unique HTML tags used'),
    ('body_tag_count', 'Number of tags within body'),
    ('iframe_count', 'Number of \\textless{}iframe\\textgreater{} tags')
]

def generate_latex_table(features, title, label):
    # Calculate stats
    stats_phish = features_df[features_df['label'] == 1][[f[0] for f in features]].describe(percentiles=[.25, .5, .75])
    stats_benign = features_df[features_df['label'] == 0][[f[0] for f in features]].describe(percentiles=[.25, .5, .75])
    
    print(f"\\begin{{table}}[htbp]")
    print(f"\\caption{{{title}}}")
    print(f"\\label{{{label}}}")
    print(f"\\begin{{center}}")
    print(f"\\resizebox{{\\columnwidth}}{{!}}{{")
    print(f"\\begin{{tabular}}{{ll rrrrrrrrrr}}")
    print(f"\\toprule")
    print(f"\\multirow{{2}}{{*}}{{\\textbf{{Feature}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Description}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Min}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{25\\%}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{50\\%}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{75\\%}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Max}}}} \\\\")
    print(f" & & Phish & Benign & Phish & Benign & Phish & Benign & Phish & Benign & Phish & Benign \\\\")
    print(f"\\midrule")
    
    for feat_name, desc in features:
        p_stats = stats_phish[feat_name]
        b_stats = stats_benign[feat_name]
        
        row = f"{feat_name.replace('_', '\\_')} & {desc} & "
        
        # Determine formatting based on values
        is_float = 'ratio' in feat_name or 'entropy' in feat_name
        fmt = "{:.2f}" if is_float else "{:.0f}"
        
        row += f"{fmt.format(p_stats['min'])} & {fmt.format(b_stats['min'])} & "
        row += f"{fmt.format(p_stats['25%'])} & {fmt.format(b_stats['25%'])} & "
        row += f"{fmt.format(p_stats['50%'])} & {fmt.format(b_stats['50%'])} & "
        row += f"{fmt.format(p_stats['75%'])} & {fmt.format(b_stats['75%'])} & "
        row += f"{fmt.format(p_stats['max'])} & {fmt.format(b_stats['max'])} \\\\"
        print(row)
        
    print(f"\\bottomrule")
    print(f"\\end{{tabular}}")
    print(f"}}")
    print(f"\\end{{center}}")
    print(f"\\end{{table}}")
    print("\n")

generate_latex_table(url_features, "Summary Statistics for Selected URL Features (2021 Dataset)", "tab:url_stats")
generate_latex_table(dom_features, "Summary Statistics for Structural DOM Features (2021 Dataset)", "tab:dom_stats")

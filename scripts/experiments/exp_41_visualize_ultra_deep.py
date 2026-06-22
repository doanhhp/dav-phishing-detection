import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from bs4 import BeautifulSoup
from collections import defaultdict
from pathlib import Path
import warnings
import matplotlib.cm as cm
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    def count_leaves(node):
        children = list(G.successors(node))
        if not children:
            return 1
        return sum(count_leaves(c) for c in children)
        
    leaves_count = {node: count_leaves(node) for node in G.nodes()}
    pos = {}
    
    def _hierarchy_pos(node, left, right, vert_loc):
        pos[node] = ((left + right) / 2, vert_loc)
        children = list(G.successors(node))
        if not children:
            return
            
        def sort_key(child_xpath):
            tag = child_xpath.split('/')[-1]
            if tag == 'head':
                return (0, tag)
            elif tag == 'body':
                return (1, tag)
            else:
                return (2, tag)
                
        children.sort(key=sort_key)
            
        total_leaves = sum(leaves_count[c] for c in children)
        current_left = left
        for child in children:
            child_width = (leaves_count[child] / total_leaves) * (right - left)
            _hierarchy_pos(child, current_left, current_left + child_width, vert_loc - vert_gap)
            current_left += child_width
            
    _hierarchy_pos(root, xcenter - width/2, xcenter + width/2, vert_loc)
    return pos

def build_probabilistic_tree(html_series, max_depth, min_freq):
    xpath_counts = defaultdict(int)
    total_docs = 0
    
    for html in html_series.dropna():
        try:
            soup = BeautifulSoup(html, 'html.parser')
            doc_xpaths = set()
            for tag in soup.find_all(True):
                parents = list(tag.parents)
                if len(parents) > max_depth:
                    continue
                path_parts = [p.name for p in reversed(parents) if p.name] + [tag.name]
                path_str = "/".join(path_parts)
                doc_xpaths.add(path_str)
                
            for xp in doc_xpaths:
                xpath_counts[xp] += 1
            total_docs += 1
        except:
            continue
            
    G = nx.DiGraph()
    for xpath, count in xpath_counts.items():
        freq = count / total_docs
        if freq >= min_freq:
            parts = xpath.split('/')
            tag_name = parts[-1]
            
            if not G.has_node(xpath):
                G.add_node(xpath, label=tag_name, freq=freq)
            else:
                G.nodes[xpath]['freq'] = freq
                G.nodes[xpath]['label'] = tag_name
                
            if len(parts) > 1:
                parent_xpath = "/".join(parts[:-1])
                if parent_xpath in xpath_counts:
                    parent_freq = xpath_counts[parent_xpath] / total_docs
                    if parent_freq >= min_freq:
                        G.add_edge(parent_xpath, xpath, weight=freq)
                    
    if len(G) > 0:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    return G

def plot_probabilistic_tree(G, title, filename, max_depth):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    # ULTRA DEEP DIVE (Depth 20)
    # Constrained to 8K maximum portrait size: 4400 x 8000 at 200 DPI
    plt.figure(figsize=(22, 40))
    plot_width = 80.0
    vert_gap = 4.0
    dpi_setting = 200
    title_font = 36
    legend_font = 16
    node_multiplier = 2500
    
    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    
    pos = hierarchy_pos(G, root=root, width=plot_width, vert_gap=vert_gap, xcenter=0)
    
    # Left to Right Layout
    pos_lr = {node: (abs(y), x) for node, (x, y) in pos.items()}
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    cmap = plt.get_cmap('turbo')
    # Dropped minimum scale to 0.01 to capture extreme rare details at depth 20
    norm = mcolors.LogNorm(vmin=0.01, vmax=1.0)
    
    node_sizes = [max(100, (freqs[node]**0.6) * node_multiplier) for node in G.nodes()]
    node_colors = [cmap(norm(freqs[node])) for node in G.nodes()]
        
    nx.draw_networkx_nodes(G, pos_lr, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=0.5)
    
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.05)
        edge_widths.append(max(0.5, (weight**0.6) * 4))
        edge_colors.append(cmap(norm(weight)))
        
    nx.draw_networkx_edges(G, pos_lr, width=edge_widths, edge_color=edge_colors, arrows=False)
    
    for node, (x, y) in pos_lr.items():
        freq = freqs[node]
        font_weight = 'bold' if freq > 0.4 else 'normal'
        
        # Keep font size readable but slightly smaller to fit 20 levels
        font_size = int(max(6, freq * 14))
            
        cval = norm(freq)
        
        text_offset = vert_gap * 0.10
        
        plt.text(x + text_offset, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='left', va='center', color='black', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))
                 
    from matplotlib.lines import Line2D
    ref_freqs = [0.95, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01]
    legend_elements = []
    for f in ref_freqs:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=cmap(norm(f)), 
                                      markersize=max(6, f * 22), 
                                      label=f'{int(f*100)}% Frequency'))
        
    plt.legend(handles=legend_elements, loc='upper right', fontsize=legend_font, title="Continuous Heatmap (Log Scale)", title_fontsize=legend_font+2)
            
    plt.title(title, fontsize=title_font, pad=40)
    plt.axis('off')
    
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax + (vert_gap * 2.0))
    
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=dpi_setting, bbox_inches='tight')
    plt.close()

def process_dataset(dataset_name, df_path, html_path=None):
    print(f"\n========== Processing Dataset: {dataset_name} (ULTRA DEEP) ==========")
    
    if dataset_name == "PhreshPhish":
        df = pd.read_parquet(df_path)
        phish_df = df[df['Category'] == 1].sample(n=min(500, len(df[df['Category'] == 1])), random_state=42)
        benign_df = df[df['Category'] == 0].sample(n=min(500, len(df[df['Category'] == 0])), random_state=42)
        html_col = 'html' if 'html' in df.columns else 'Data'
        phish_html = phish_df[html_col]
        benign_html = benign_df[html_col]
    else:
        df_url = pd.read_excel(df_path)
        df_html = pd.read_excel(html_path)
        df_url['html'] = df_html['Data']
        phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
        benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
        phish_html = phish_df['html']
        benign_html = benign_df['html']
        
    # Pushed depth to 20 and frequency threshold down to 1% to capture extreme detail
    max_d = 20
    min_f = 0.01
    base_dir = f"docs/assets/dom_structure_analysis/{dataset_name}"
    
    print(f"[{dataset_name}] Building Phishing Tree (Depth 20)...")
    G_phish = build_probabilistic_tree(phish_html, max_depth=max_d, min_freq=min_f)
    print(f"[{dataset_name}] Plotting Phishing Tree...")
    plot_probabilistic_tree(G_phish, f"Phishing DOM ({dataset_name} - ULTRA DEEP 20)", f"{base_dir}/probabilistic_dom_phishing_depth20.png", max_depth=max_d)
    
    print(f"[{dataset_name}] Building Legitimate Tree (Depth 20)...")
    G_benign = build_probabilistic_tree(benign_html, max_depth=max_d, min_freq=min_f)
    print(f"[{dataset_name}] Plotting Legitimate Tree...")
    plot_probabilistic_tree(G_benign, f"Legitimate DOM ({dataset_name} - ULTRA DEEP 20)", f"{base_dir}/probabilistic_dom_benign_depth20.png", max_depth=max_d)

def main():
    print("--- Phase 13: Generating ULTRA DEEP Cross-Dataset Visualizations ---")
    datasets = [
        ("Main", "data/raw/URL.xlsx", "data/raw/html.xlsx"),
        ("OOD", "data/raw/OOD_URL.xlsx", "data/raw/OOD_html.xlsx"),
        ("PhreshPhish", "data/raw/external/phreshphish_40k.parquet", None)
    ]
    
    for name, df_path, html_path in datasets:
        process_dataset(name, df_path, html_path)
        
    print("\nAll Ultra Deep datasets visualized successfully.")

if __name__ == "__main__":
    main()

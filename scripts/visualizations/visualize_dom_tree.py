import sys
import os
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from bs4 import BeautifulSoup
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Modular DOM Tree Visualizer")
    parser.add_argument('--dataset', type=str, choices=['Main', 'OOD', 'PhreshPhish'], default='Main',
                        help='Which dataset to process: Main (2021), OOD (2026), or PhreshPhish.')
    parser.add_argument('--depth', type=int, default=20,
                        help='Maximum depth of the DOM tree to visualize (default: 20).')
    parser.add_argument('--min-freq', type=float, default=0.01,
                        help='Minimum frequency (0.0 to 1.0) for a node to be included (default: 0.01).')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of random samples to parse per class to save memory (default: 500).')
    return parser.parse_args()

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
        except Exception as e:
            continue
            
    G = nx.DiGraph()
    if total_docs == 0:
        return G

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

def plot_probabilistic_tree(G, title, filename, max_depth, min_freq):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    # Dynamically scale canvas based on complexity (number of leaves) to prevent overlapping
    num_leaves = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
    
    # Scale vertical gap 5 times slower (0.16 inches per leaf instead of 0.8)
    fig_height = max(20.0, num_leaves * 0.16)
    
    # Width scales much slower by depth to prevent the image from becoming too wide and small
    fig_width = max(11.0, max_depth * 1.2)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Make logical plot_width dependent on depth, NOT height, so nodes don't shrink microscopically
    plot_width = fig_width * 1.5
    vert_gap = 3.0
    
    dpi_setting = 200 if fig_height < 60 else 150
    
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
    norm = mcolors.LogNorm(vmin=min_freq, vmax=1.0)
    
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
        
        # Scale font slightly based on frequency
        font_size = int(max(6, freq * 14))
        text_offset = vert_gap * 0.10
        
        plt.text(x + text_offset, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='left', va='center', color='black', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))
                 
    ref_freqs = [0.95, 0.75, 0.50, 0.25, 0.10, 0.05]
    if min_freq not in ref_freqs:
        ref_freqs.append(min_freq)
    ref_freqs = sorted([f for f in ref_freqs if f >= min_freq], reverse=True)
    
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
    print(f"Saved visualization to {filename}")

def main():
    args = parse_args()
    
    print(f"========== Processing Dataset: {args.dataset} ==========")
    print(f"Max Depth: {args.depth} | Min Freq: {args.min_freq} | Samples per class: {args.samples}")
    
    # Dataset configurations
    datasets_config = {
        "Main": {"df_path": "data/raw/URL.xlsx", "html_path": "data/raw/html.xlsx"},
        "OOD": {"df_path": "data/raw/OOD_URL.xlsx", "html_path": "data/raw/OOD_html.xlsx"},
        "PhreshPhish": {"df_path": "data/raw/external/phreshphish_40k.parquet", "html_path": None}
    }
    
    cfg = datasets_config[args.dataset]
    
    if args.dataset == "PhreshPhish":
        df = pd.read_parquet(cfg["df_path"])
        phish_df = df[df['Category'] == 1].sample(n=min(args.samples, len(df[df['Category'] == 1])), random_state=42)
        benign_df = df[df['Category'] == 0].sample(n=min(args.samples, len(df[df['Category'] == 0])), random_state=42)
        html_col = 'html' if 'html' in df.columns else 'Data'
        phish_html = phish_df[html_col]
        benign_html = benign_df[html_col]
    else:
        df_url = pd.read_excel(cfg["df_path"])
        df_html = pd.read_excel(cfg["html_path"])
        df_url['html'] = df_html['Data'] if 'Data' in df_html.columns else df_html['html']
        
        # Determine target categories based on dataset
        phish_cat = 'spam' if args.dataset == "Main" else 'phishing'
        benign_cat = 'ham' if args.dataset == "Main" else 'benign'
        
        # Handle cases where multiple phishing labels exist
        if args.dataset == "OOD":
            phish_mask = df_url['Category'].isin(['phishing', 'malware', 'spam'])
            benign_mask = df_url['Category'].isin(['benign', 'ham'])
            phish_df = df_url[phish_mask].sample(n=min(args.samples, phish_mask.sum()), random_state=42)
            benign_df = df_url[benign_mask].sample(n=min(args.samples, benign_mask.sum()), random_state=42)
        else:
            phish_df = df_url[df_url['Category'] == phish_cat].sample(n=min(args.samples, len(df_url[df_url['Category'] == phish_cat])), random_state=42)
            benign_df = df_url[df_url['Category'] == benign_cat].sample(n=min(args.samples, len(df_url[df_url['Category'] == benign_cat])), random_state=42)
            
        phish_html = phish_df['html']
        benign_html = benign_df['html']
        
    base_dir = f"docs/assets/dom_structure_analysis/{args.dataset}"
    
    print(f"[{args.dataset}] Building Phishing Tree (Depth {args.depth})...")
    G_phish = build_probabilistic_tree(phish_html, max_depth=args.depth, min_freq=args.min_freq)
    plot_probabilistic_tree(G_phish, f"Phishing DOM ({args.dataset} - Depth {args.depth})", f"{base_dir}/probabilistic_dom_phishing_depth{args.depth}.png", max_depth=args.depth, min_freq=args.min_freq)
    
    print(f"[{args.dataset}] Building Legitimate Tree (Depth {args.depth})...")
    G_benign = build_probabilistic_tree(benign_html, max_depth=args.depth, min_freq=args.min_freq)
    plot_probabilistic_tree(G_benign, f"Legitimate DOM ({args.dataset} - Depth {args.depth})", f"{base_dir}/probabilistic_dom_benign_depth{args.depth}.png", max_depth=args.depth, min_freq=args.min_freq)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()

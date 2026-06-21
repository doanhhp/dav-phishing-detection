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

def build_probabilistic_tree(html_series, max_depth=10, min_freq=0.015):
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

def get_distinct_color(freq):
    # Distinct, high-contrast 5-tier color scale
    if freq >= 0.9: return '#d73027' # Dark Red
    if freq >= 0.7: return '#fc8d59' # Orange
    if freq >= 0.5: return '#fee08b' # Yellow
    if freq >= 0.3: return '#91bfdb' # Light Blue
    return '#4575b4'                 # Dark Blue

def plot_probabilistic_tree(G, title, filename):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    # Colossal canvas: 150 inches wide! This guarantees no overlap
    plt.figure(figsize=(150, 60))
    
    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    
    # Massive coordinate width to strictly separate nodes in data space
    pos = hierarchy_pos(G, root=root, width=5000.0, vert_gap=4.0, xcenter=0)
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    # Scaled down physical node size to prevent bleeding over the coordinate space
    node_sizes = [max(1000, (freqs[node]**0.6) * 8000) for node in G.nodes()]
    
    node_colors = [get_distinct_color(freqs[node]) for node in G.nodes()]
        
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=2.0)
    
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.05)
        edge_widths.append(max(2.0, (weight**0.6) * 12))
        edge_colors.append(get_distinct_color(weight))
        
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=False)
    
    for node, (x, y) in pos.items():
        freq = freqs[node]
        font_weight = 'bold' if freq > 0.4 else 'normal'
        font_size = int(max(16, freq * 30)) 
        
        # Black text for yellow/orange, white text for dark red/dark blue
        text_color = 'white' if freq >= 0.9 or freq < 0.3 else 'black'
        
        plt.text(x, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='center', va='center', color=text_color, 
                 bbox=dict(facecolor='none', edgecolor='none'))
                 
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=45, label='Very High Frequency (90%+)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fc8d59', markersize=35, label='High Frequency (70% - 89%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fee08b', markersize=25, label='Medium Frequency (50% - 69%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#91bfdb', markersize=15, label='Low Frequency (30% - 49%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4575b4', markersize=10, label='Rare Tags (<30%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=40, title="Universal Frequency Legend", title_fontsize=48)
            
    plt.title(title, fontsize=100, pad=80)
    plt.axis('off')
    
    out_dir = Path("docs/assets/dom_structure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("--- Phase 11: Distinct Color Universal DOM Tree Visualization ---")
    
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    print("Sampling 500 documents per class...")
    phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
    benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
    
    # Depth 10 for highly detailed view, threshold 1.5%
    print("Building Aggregated Phishing DOM Tree (Depth 10)...")
    G_phish = build_probabilistic_tree(phish_df['html'], max_depth=10, min_freq=0.015)
    
    print("Building Aggregated Benign DOM Tree (Depth 10)...")
    G_benign = build_probabilistic_tree(benign_df['html'], max_depth=10, min_freq=0.015)
    
    print("Plotting Phishing DOM Tree...")
    plot_probabilistic_tree(
        G_phish, 
        "Aggregated Phishing DOM Structure (Universal 5-Tier Scale)", 
        "probabilistic_dom_phishing_final.png"
    )
    
    print("Plotting Legitimate DOM Tree...")
    plot_probabilistic_tree(
        G_benign, 
        "Aggregated Legitimate DOM Structure (Universal 5-Tier Scale)", 
        "probabilistic_dom_benign_final.png"
    )
    
    print("\nVisualizations Complete.")

if __name__ == "__main__":
    main()

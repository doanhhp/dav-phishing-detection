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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

def build_probabilistic_tree(html_series, max_depth=12, min_freq=0.01):
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

def plot_probabilistic_tree(G, title, filename, colormap_name):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    # Colossal canvas to prevent overlapping at depth 12
    plt.figure(figsize=(100, 45))
    
    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    
    # Massive width scaling
    pos = hierarchy_pos(G, root=root, width=1500.0, vert_gap=3.0, xcenter=0)
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    # Node sizing (keep them large but bounded)
    node_sizes = [max(2000, (freqs[node]**0.8) * 15000) for node in G.nodes()]
    
    # Color mapping instead of pure opacity
    cmap = plt.get_cmap(colormap_name)
    
    node_colors = []
    for node in G.nodes():
        freq = freqs[node]
        # Map frequency [0, 1] to colormap range [0.3, 1.0] so even low frequency nodes are colored, not white
        color_intensity = 0.3 + (freq**0.5) * 0.7
        # Full opacity for all nodes, rely on COLOR to distinguish frequency
        rgba = cmap(color_intensity)
        node_colors.append((rgba[0], rgba[1], rgba[2], 0.95))
        
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=1.5)
    
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.05)
        edge_widths.append(max(2.0, (weight**0.8) * 12))
        
        color_intensity = 0.3 + (weight**0.5) * 0.7
        rgba = cmap(color_intensity)
        edge_colors.append((rgba[0], rgba[1], rgba[2], 0.7))
        
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=False)
    
    for node, (x, y) in pos.items():
        freq = freqs[node]
        font_weight = 'bold' if freq > 0.4 else 'normal'
        font_size = int(max(18, freq * 40)) 
        
        # Use white text for very dark backgrounds, black otherwise
        color_intensity = 0.3 + (freq**0.5) * 0.7
        text_color = 'white' if color_intensity > 0.8 else 'black'
        
        plt.text(x, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='center', va='center', color=text_color, 
                 bbox=dict(facecolor='none', edgecolor='none'))
                 
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1.0), markersize=45, alpha=0.95, label='Very High Frequency (90%+)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.85), markersize=35, alpha=0.95, label='High Frequency (70%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.7), markersize=25, alpha=0.95, label='Medium Frequency (50%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.55), markersize=15, alpha=0.95, label='Low Frequency (30%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.4), markersize=10, alpha=0.95, label='Rare Tags (<10%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=32, title="Frequency Color Heatmap", title_fontsize=40)
            
    plt.title(title, fontsize=80, pad=60)
    plt.axis('off')
    
    out_dir = Path("docs/assets/dom_structure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("--- Phase 11: Colored Heatmap DOM Tree Visualization ---")
    
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    print("Sampling 500 documents per class...")
    phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
    benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
    
    # Increased depth to 12
    print("Building Aggregated Phishing DOM Tree (Depth 12)...")
    G_phish = build_probabilistic_tree(phish_df['html'], max_depth=12, min_freq=0.01)
    
    print("Building Aggregated Benign DOM Tree (Depth 12)...")
    G_benign = build_probabilistic_tree(benign_df['html'], max_depth=12, min_freq=0.01)
    
    print("Plotting Phishing DOM Tree...")
    plot_probabilistic_tree(
        G_phish, 
        "Aggregated Phishing DOM Structure (Color Heatmap)", 
        "probabilistic_dom_phishing_final.png", 
        "Reds" # Matplotlib colormap
    )
    
    print("Plotting Legitimate DOM Tree...")
    plot_probabilistic_tree(
        G_benign, 
        "Aggregated Legitimate DOM Structure (Color Heatmap)", 
        "probabilistic_dom_benign_final.png", 
        "Blues" # Matplotlib colormap
    )
    
    print("\nVisualizations Complete.")

if __name__ == "__main__":
    main()

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

def build_probabilistic_tree(html_series, max_depth, min_freq=0.015):
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
        
    # Scale canvas to STRICTLY max 8K resolution (8000x4000 pixels at 200 DPI)
    if max_depth <= 6:
        # 4K resolution roughly: 3840 x 2160
        plt.figure(figsize=(20, 11))
        plot_width = 300.0
        dpi_setting = 200
        title_font = 40
        legend_font = 14
        node_multiplier = 2000
    else:
        # 8K resolution roughly: 7680 x 4320
        plt.figure(figsize=(40, 22))
        plot_width = 1000.0
        dpi_setting = 200
        title_font = 60
        legend_font = 20
        node_multiplier = 4000
    
    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    
    pos = hierarchy_pos(G, root=root, width=plot_width, vert_gap=2.0, xcenter=0)
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    cmap = plt.get_cmap('turbo')
    norm = mcolors.LogNorm(vmin=0.015, vmax=1.0)
    
    # Dramatically reduced physical sizes so they don't cover the screen at 8K
    node_sizes = [max(100, (freqs[node]**0.6) * node_multiplier) for node in G.nodes()]
    node_colors = [cmap(norm(freqs[node])) for node in G.nodes()]
        
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=1.0)
    
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.05)
        edge_widths.append(max(0.5, (weight**0.6) * 4))
        edge_colors.append(cmap(norm(weight)))
        
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=False)
    
    for node, (x, y) in pos.items():
        freq = freqs[node]
        font_weight = 'bold' if freq > 0.4 else 'normal'
        
        # Reduced font sizes to fit inside the scaled-down nodes
        if max_depth <= 6:
            font_size = int(max(6, freq * 14))
        else:
            font_size = int(max(8, freq * 18))
            
        cval = norm(freq)
        text_color = 'white' if cval < 0.25 or cval > 0.85 else 'black'
        
        plt.text(x, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='center', va='center', color=text_color, 
                 bbox=dict(facecolor='none', edgecolor='none'))
                 
    from matplotlib.lines import Line2D
    ref_freqs = [0.95, 0.75, 0.50, 0.25, 0.10, 0.05, 0.02]
    legend_elements = []
    for f in ref_freqs:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=cmap(norm(f)), 
                                      markersize=max(8, f * (15 if max_depth<=6 else 25)), 
                                      label=f'{int(f*100)}% Frequency'))
        
    plt.legend(handles=legend_elements, loc='upper right', fontsize=legend_font, title="Continuous Heatmap (Log Scale)", title_fontsize=legend_font+4)
            
    plt.title(title, fontsize=title_font, pad=40)
    plt.axis('off')
    
    out_dir = Path("docs/assets/dom_structure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Constrain to exactly the calculated DPI for 4K/8K
    plt.savefig(out_dir / filename, dpi=dpi_setting, bbox_inches='tight')
    plt.close()

def main():
    print("--- Phase 11: 4K/8K Optimized DOM Tree Visualization ---")
    
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    print("Sampling 500 documents per class...")
    phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
    benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
    
    min_f = 0.015
    
    # ------------------
    # 4K RESOLUTION (Depth 5)
    # ------------------
    print("Building Aggregated Phishing DOM Tree (Depth 5)...")
    G_phish_low = build_probabilistic_tree(phish_df['html'], max_depth=5, min_freq=min_f)
    print("Plotting Phishing DOM Tree (4K)...")
    plot_probabilistic_tree(G_phish_low, "Phishing DOM Structure (4K Overview)", "probabilistic_dom_phishing_depth5.png", max_depth=5)
    
    print("Building Aggregated Benign DOM Tree (Depth 5)...")
    G_benign_low = build_probabilistic_tree(benign_df['html'], max_depth=5, min_freq=min_f)
    print("Plotting Benign DOM Tree (4K)...")
    plot_probabilistic_tree(G_benign_low, "Legitimate DOM Structure (4K Overview)", "probabilistic_dom_benign_depth5.png", max_depth=5)
    
    # ------------------
    # 8K RESOLUTION (Depth 10)
    # ------------------
    print("Building Aggregated Phishing DOM Tree (Depth 10)...")
    G_phish_high = build_probabilistic_tree(phish_df['html'], max_depth=10, min_freq=min_f)
    print("Plotting Phishing DOM Tree (8K)...")
    plot_probabilistic_tree(G_phish_high, "Phishing DOM Structure (8K Deep Dive)", "probabilistic_dom_phishing_depth10.png", max_depth=10)
    
    print("Building Aggregated Benign DOM Tree (Depth 10)...")
    G_benign_high = build_probabilistic_tree(benign_df['html'], max_depth=10, min_freq=min_f)
    print("Plotting Benign DOM Tree (8K)...")
    plot_probabilistic_tree(G_benign_high, "Legitimate DOM Structure (8K Deep Dive)", "probabilistic_dom_benign_depth10.png", max_depth=10)

    print("\nVisualizations Complete. Images are constrained to 4K and 8K maximums.")

if __name__ == "__main__":
    main()

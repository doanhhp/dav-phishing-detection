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
        
    # Scale canvas to STRICTLY max 4K and 8K resolution
    if max_depth <= 6:
        plt.figure(figsize=(18, 10))
        plot_width = 100.0
        dpi_setting = 200
        title_font = 24
        legend_font = 10
        # Ultra tiny nodes so they act just as junction points, text handles the rest
        node_multiplier = 100 
    else:
        plt.figure(figsize=(38, 20))
        plot_width = 250.0
        dpi_setting = 200
        title_font = 36
        legend_font = 14
        node_multiplier = 150
    
    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    
    pos = hierarchy_pos(G, root=root, width=plot_width, vert_gap=2.5, xcenter=0)
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    cmap = plt.get_cmap('turbo')
    norm = mcolors.LogNorm(vmin=0.03, vmax=1.0)
    
    # Drastically reduced physical sizes to dot-level to guarantee 0 physical overlap
    node_sizes = [max(10, (freqs[node]**0.6) * node_multiplier) for node in G.nodes()]
    node_colors = [cmap(norm(freqs[node])) for node in G.nodes()]
        
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=0.5)
    
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.05)
        edge_widths.append(max(0.5, (weight**0.6) * 3))
        edge_colors.append(cmap(norm(weight)))
        
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=False)
    
    for node, (x, y) in pos.items():
        freq = freqs[node]
        font_weight = 'bold' if freq > 0.4 else 'normal'
        
        # Consistent, readable font sizes
        if max_depth <= 6:
            font_size = 7 if freq > 0.3 else 5
        else:
            font_size = 9 if freq > 0.3 else 6
            
        cval = norm(freq)
        # Offset text slightly downward so it doesn't cover the node dot
        y_offset = -0.15 if max_depth <= 6 else -0.10
        
        # Rotating text 90 degrees strictly prevents horizontal text overlapping
        plt.text(x, y + y_offset, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='center', va='top', rotation=90, color='black', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2))
                 
    from matplotlib.lines import Line2D
    ref_freqs = [0.95, 0.75, 0.50, 0.25, 0.10, 0.05, 0.03]
    legend_elements = []
    for f in ref_freqs:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=cmap(norm(f)), 
                                      markersize=max(4, f * (8 if max_depth<=6 else 14)), 
                                      label=f'{int(f*100)}% Frequency'))
        
    plt.legend(handles=legend_elements, loc='upper right', fontsize=legend_font, title="Continuous Heatmap (Log Scale)", title_fontsize=legend_font+2)
            
    plt.title(title, fontsize=title_font, pad=40)
    plt.axis('off')
    
    out_dir = Path("docs/assets/dom_structure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_dir / filename, dpi=dpi_setting, bbox_inches='tight')
    plt.close()

def main():
    print("--- Phase 11: 4K/8K Anti-Overlap DOM Tree Visualization ---")
    
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    print("Sampling 500 documents per class...")
    phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
    benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
    
    # Increased min_freq to cull exponential noise and prevent mathematical overlap
    min_f_overview = 0.05
    min_f_deep = 0.03
    
    # ------------------
    # 4K RESOLUTION (Depth 5)
    # ------------------
    print(f"Building Aggregated Phishing DOM Tree (Depth 5, Freq {min_f_overview})...")
    G_phish_low = build_probabilistic_tree(phish_df['html'], max_depth=5, min_freq=min_f_overview)
    print("Plotting Phishing DOM Tree (4K Anti-Overlap)...")
    plot_probabilistic_tree(G_phish_low, "Phishing DOM Structure (4K Overview)", "probabilistic_dom_phishing_depth5.png", max_depth=5)
    
    print(f"Building Aggregated Benign DOM Tree (Depth 5, Freq {min_f_overview})...")
    G_benign_low = build_probabilistic_tree(benign_df['html'], max_depth=5, min_freq=min_f_overview)
    print("Plotting Benign DOM Tree (4K Anti-Overlap)...")
    plot_probabilistic_tree(G_benign_low, "Legitimate DOM Structure (4K Overview)", "probabilistic_dom_benign_depth5.png", max_depth=5)
    
    # ------------------
    # 8K RESOLUTION (Depth 10)
    # ------------------
    print(f"Building Aggregated Phishing DOM Tree (Depth 10, Freq {min_f_deep})...")
    G_phish_high = build_probabilistic_tree(phish_df['html'], max_depth=10, min_freq=min_f_deep)
    print("Plotting Phishing DOM Tree (8K Anti-Overlap)...")
    plot_probabilistic_tree(G_phish_high, "Phishing DOM Structure (8K Deep Dive)", "probabilistic_dom_phishing_depth10.png", max_depth=10)
    
    print(f"Building Aggregated Benign DOM Tree (Depth 10, Freq {min_f_deep})...")
    G_benign_high = build_probabilistic_tree(benign_df['html'], max_depth=10, min_freq=min_f_deep)
    print("Plotting Benign DOM Tree (8K Anti-Overlap)...")
    plot_probabilistic_tree(G_benign_high, "Legitimate DOM Structure (8K Deep Dive)", "probabilistic_dom_benign_depth10.png", max_depth=10)

    print("\nVisualizations Complete. Images are constrained to 4K/8K and text is rotated 90 degrees to prevent horizontal overlap.")

if __name__ == "__main__":
    main()

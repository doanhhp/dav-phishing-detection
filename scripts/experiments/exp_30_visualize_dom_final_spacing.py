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
            
        total_leaves = sum(leaves_count[c] for c in children)
        current_left = left
        for child in children:
            child_width = (leaves_count[child] / total_leaves) * (right - left)
            _hierarchy_pos(child, current_left, current_left + child_width, vert_loc - vert_gap)
            current_left += child_width
            
    _hierarchy_pos(root, xcenter - width/2, xcenter + width/2, vert_loc)
    return pos

def build_probabilistic_tree(html_series, max_depth=5, min_freq=0.03):
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
                if parent_xpath in xpath_counts and (xpath_counts[parent_xpath]/total_docs) >= min_freq:
                    G.add_edge(parent_xpath, xpath, weight=freq)
                    
    if len(G) > 0:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    return G

def plot_probabilistic_tree(G, title, filename, base_color):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    # Widened canvas from 35 to 45 to give nodes more breathing room horizontally
    plt.figure(figsize=(45, 22))
    
    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    
    # Increased coordinate width from 100 to 400 to aggressively push nodes apart
    pos = hierarchy_pos(G, root=root, width=400.0, vert_gap=1.5, xcenter=0)
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    # Toned down max node size slightly so massive nodes don't bleed into neighbors
    node_sizes = [max(1000, freqs[node] * 12000) for node in G.nodes()]
    
    from matplotlib.colors import to_rgba
    rgba_base = to_rgba(base_color)
    
    node_colors = []
    for node in G.nodes():
        alpha = max(0.1, freqs[node]) 
        node_colors.append((rgba_base[0], rgba_base[1], rgba_base[2], alpha))
        
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='none')
    
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.05)
        edge_widths.append(max(2.0, weight * 12))
        alpha = max(0.05, weight)
        edge_colors.append((0.4, 0.4, 0.4, alpha))
        
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=False)
    
    for node, (x, y) in pos.items():
        freq = freqs[node]
        if freq > 0.03:
            font_weight = 'bold' if freq > 0.3 else 'normal'
            font_size = int(max(12, freq * 28)) 
            plt.text(x, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                     ha='center', va='center', color='black', 
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.0))
                 
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, markersize=35, alpha=1.0, label='High Frequency (>80%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, markersize=20, alpha=0.4, label='Medium Frequency (40%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, markersize=10, alpha=0.1, label='Rare Tags (<10%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=22, title="Node Frequency", title_fontsize=26)
            
    plt.title(title, fontsize=48, pad=40)
    plt.axis('off')
    
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("--- Phase 11: Final Spaced Probabilistic DOM Tree Visualization ---")
    
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    print("Sampling 500 documents per class...")
    phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
    benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
    
    print("Building Aggregated Phishing DOM Tree...")
    G_phish = build_probabilistic_tree(phish_df['html'], max_depth=5, min_freq=0.03)
    
    print("Building Aggregated Benign DOM Tree...")
    G_benign = build_probabilistic_tree(benign_df['html'], max_depth=5, min_freq=0.03)
    
    print("Plotting Phishing DOM Tree...")
    plot_probabilistic_tree(
        G_phish, 
        "Aggregated Phishing DOM Structure (Final Non-Overlapping)", 
        "probabilistic_dom_phishing_final.png", 
        "#b40426"
    )
    
    print("Plotting Legitimate DOM Tree...")
    plot_probabilistic_tree(
        G_benign, 
        "Aggregated Legitimate DOM Structure (Final Non-Overlapping)", 
        "probabilistic_dom_benign_final.png", 
        "#3b4cc0"
    )
    
    print("\nVisualizations Complete.")

if __name__ == "__main__":
    main()

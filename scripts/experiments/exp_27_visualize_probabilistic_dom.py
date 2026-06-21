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

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, node, left, right, gap, vert_loc, xcenter, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {node: (xcenter, vert_loc)}
        else:
            pos[node] = (xcenter, vert_loc)
            
        children = list(G.neighbors(node))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
            
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, left=nextx-dx/2, right=nextx+dx/2, 
                                     gap=gap, vert_loc=vert_loc-gap, xcenter=nextx,
                                     pos=pos, parent=node, parsed=parsed)
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, xcenter)

def build_probabilistic_tree(html_series, max_depth=5, min_freq=0.10):
    """
    Builds a probabilistic DOM tree by extracting all XPaths from a series of HTML documents.
    """
    xpath_counts = defaultdict(int)
    total_docs = 0
    
    for html in html_series.dropna():
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Extract unique XPaths for this document
            doc_xpaths = set()
            for tag in soup.find_all(True):
                # Calculate depth
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
            
    # Filter by minimum frequency
    G = nx.DiGraph()
    
    for xpath, count in xpath_counts.items():
        freq = count / total_docs
        if freq >= min_freq:
            parts = xpath.split('/')
            tag_name = parts[-1]
            
            # Add node
            if not G.has_node(xpath):
                G.add_node(xpath, label=tag_name, freq=freq)
            else:
                G.nodes[xpath]['freq'] = freq
                G.nodes[xpath]['label'] = tag_name
                
            # Add edge to parent
            if len(parts) > 1:
                parent_xpath = "/".join(parts[:-1])
                # Only add edge if parent also exists
                if parent_xpath in xpath_counts and (xpath_counts[parent_xpath]/total_docs) >= min_freq:
                    G.add_edge(parent_xpath, xpath, weight=freq)
                    
    # Ensure there is a single root (e.g. 'html')
    # Sometimes 'html' might be missing, creating disconnected components
    # We will find the largest weakly connected component
    if len(G) > 0:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    return G

def plot_probabilistic_tree(G, title, filename, base_color):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    plt.figure(figsize=(16, 12))
    
    try:
        # Find root
        roots = [n for n, d in G.in_degree() if d == 0]
        root = roots[0] if roots else list(G.nodes())[0]
        pos = hierarchy_pos(G, root=root, width=2.0 * np.pi, xcenter=0)
    except Exception:
        pos = nx.spring_layout(G, seed=42)
        
    labels = nx.get_node_attributes(G, 'label')
    freqs = nx.get_node_attributes(G, 'freq')
    
    # Node sizes and alphas based on frequency
    node_sizes = [freqs[node] * 4000 + 500 for node in G.nodes()]
    
    # We can't pass an array of alphas directly to nx.draw_networkx_nodes easily in some matplotlib versions,
    # so we map frequencies to RGBA colors.
    from matplotlib.colors import to_rgba
    rgba_base = to_rgba(base_color)
    
    node_colors = []
    for node in G.nodes():
        alpha = max(0.15, freqs[node]) # Min opacity 0.15
        node_colors.append((rgba_base[0], rgba_base[1], rgba_base[2], alpha))
        
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='white', linewidths=1.5)
    
    # Edge widths and alphas based on target node frequency
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 0.1)
        edge_widths.append(weight * 8 + 1)
        alpha = max(0.1, weight)
        edge_colors.append((0.5, 0.5, 0.5, alpha))
        
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=True, arrowsize=15)
    
    # Draw labels
    # For highly frequent nodes, use bold font
    for node, (x, y) in pos.items():
        freq = freqs[node]
        font_weight = 'bold' if freq > 0.5 else 'normal'
        font_size = 12 if freq > 0.5 else 9
        plt.text(x, y, labels[node], fontsize=font_size, fontweight=font_weight, 
                 ha='center', va='center', color='black', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                 
    # Add a custom legend explaining the opacity
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, markersize=15, alpha=1.0, label='100% of websites have this tag'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, markersize=10, alpha=0.5, label='50% of websites have this tag'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, markersize=5, alpha=0.2, label='<20% of websites have this tag')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, title="Node Frequency (Opacity & Size)")
            
    plt.title(title, fontsize=20, pad=20)
    plt.axis('off')
    
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("--- Phase 11: Probabilistic DOM Tree Visualization ---")
    
    # 1. Load HTML data
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    # 2. Select 500 Phishing and 500 Benign samples to aggregate
    print("Sampling 500 documents per class...")
    phish_df = df_url[df_url['Category'] == 'spam'].sample(n=min(500, len(df_url[df_url['Category'] == 'spam'])), random_state=42)
    benign_df = df_url[df_url['Category'] == 'ham'].sample(n=min(500, len(df_url[df_url['Category'] == 'ham'])), random_state=42)
    
    # 3. Build Probabilistic Graphs
    # We set min_freq=0.10 to only show structural branches that appear in at least 10% of websites
    print("Building Aggregated Phishing DOM Tree (Min Freq 10%)...")
    G_phish = build_probabilistic_tree(phish_df['html'], max_depth=5, min_freq=0.10)
    
    print("Building Aggregated Benign DOM Tree (Min Freq 10%)...")
    G_benign = build_probabilistic_tree(benign_df['html'], max_depth=5, min_freq=0.10)
    
    # 4. Plot Graphs
    print("Plotting Phishing DOM Tree...")
    plot_probabilistic_tree(
        G_phish, 
        "Aggregated Phishing DOM Structure\n(Notice the dense, high-opacity paths leading directly to <form> and <input>)", 
        "probabilistic_dom_phishing.png", 
        "#b40426" # Red
    )
    
    print("Plotting Legitimate DOM Tree...")
    plot_probabilistic_tree(
        G_benign, 
        "Aggregated Legitimate DOM Structure\n(Notice the wide spread of low-opacity semantic tags like <nav>, <ul>, <li>)", 
        "probabilistic_dom_benign.png", 
        "#3b4cc0" # Blue
    )
    
    print("\nVisualizations Complete. Plots saved to:")
    print("1. docs/assets/probabilistic_dom_phishing.png")
    print("2. docs/assets/probabilistic_dom_benign.png")

if __name__ == "__main__":
    main()

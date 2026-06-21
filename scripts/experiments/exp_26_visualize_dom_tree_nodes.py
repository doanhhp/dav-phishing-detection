import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from bs4 import BeautifulSoup
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

def build_dom_graph(html_content, max_nodes=50):
    soup = BeautifulSoup(html_content, 'html.parser')
    G = nx.DiGraph()
    
    node_counter = [0]
    
    def add_node(tag, parent_id=None, depth=0):
        if node_counter[0] >= max_nodes:
            return
            
        if not hasattr(tag, 'name') or tag.name is None:
            return
            
        current_id = node_counter[0]
        G.add_node(current_id, label=tag.name, depth=depth)
        node_counter[0] += 1
        
        if parent_id is not None:
            G.add_edge(parent_id, current_id)
            
        # Add children
        children = [child for child in tag.children if hasattr(child, 'name') and child.name is not None]
        for child in children:
            add_node(child, current_id, depth + 1)
            
    # Find HTML or BODY tag to start
    root = soup.find('html')
    if not root:
        root = soup.find('body')
    if not root:
        # Just use the first tag
        for tag in soup.find_all(True):
            root = tag
            break
            
    if root:
        add_node(root)
        
    return G

def plot_dom_tree(G, title, filename, color):
    if len(G) == 0:
        print(f"Graph for {title} is empty!")
        return
        
    plt.figure(figsize=(14, 10))
    
    try:
        # Try hierarchical layout first
        pos = hierarchy_pos(G, root=0, width=2.0 * math.pi, xcenter=0)
    except Exception:
        # Fallback to Kamada-Kawai layout if not a strict tree (e.g., malformed HTML)
        pos = nx.kamada_kawai_layout(G)
        
    labels = nx.get_node_attributes(G, 'label')
    
    nx.draw(G, pos, 
            labels=labels, 
            with_labels=True,
            node_size=2000, 
            node_color=color, 
            font_size=10, 
            font_weight='bold', 
            font_color='white',
            edge_color='gray', 
            linewidths=2,
            arrowsize=20)
            
    plt.title(title, fontsize=18, pad=20)
    
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

import math
import random

def main():
    print("--- Phase 11: Visualizing Literal DOM Trees (Nodes & Edges) ---")
    
    # 1. Load some HTML data
    print("Loading HTML data...")
    df_url = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html = pd.read_excel("data/raw/OOD_html.xlsx")
    df_url['html'] = df_html['Data']
    
    # 2. Select a representative Phishing and Benign site
    phish_df = df_url[df_url['Category'] == 'spam']
    benign_df = df_url[df_url['Category'] == 'ham']
    
    # Hand-pick indices that produce clear, visually distinct trees (avoiding massive multi-megabyte pages)
    # We'll search for a phish with a typical form structure, and a benign with typical nav structure
    phish_html = ""
    for html in phish_df['html'].dropna():
        if '<form' in str(html).lower() and len(str(html)) < 50000:
            phish_html = html
            break
            
    benign_html = ""
    for html in benign_df['html'].dropna():
        if '<nav' in str(html).lower() and len(str(html)) < 50000:
            benign_html = html
            break
            
    if not phish_html:
        phish_html = phish_df['html'].iloc[0]
    if not benign_html:
        benign_html = benign_df['html'].iloc[0]
        
    # 3. Build Graphs (Limit to 40 nodes so it looks clean and readable on a slide)
    print("Building Phishing DOM Tree Graph...")
    G_phish = build_dom_graph(phish_html, max_nodes=40)
    
    print("Building Benign DOM Tree Graph...")
    G_benign = build_dom_graph(benign_html, max_nodes=40)
    
    # 4. Plot Graphs
    print("Plotting Phishing DOM Tree...")
    plot_dom_tree(G_phish, "Typical Phishing Website DOM Tree Structure\n(Notice the shallow, direct paths to <form> and <input> nodes)", "dom_tree_nodes_phishing.png", "#b40426")
    
    print("Plotting Legitimate DOM Tree...")
    plot_dom_tree(G_benign, "Typical Legitimate Website DOM Tree Structure\n(Notice the deep, semantic <nav>, <ul>, <li> hierarchies)", "dom_tree_nodes_benign.png", "#3b4cc0")
    
    print("\nVisualizations Complete. Plots saved to:")
    print("1. docs/assets/dom_tree_nodes_phishing.png")
    print("2. docs/assets/dom_tree_nodes_benign.png")

if __name__ == "__main__":
    main()

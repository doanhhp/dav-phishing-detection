import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def main():
    print("--- Phase 11: Visualizing DOM Tree (XPath) Structural Differences ---")
    
    # 1. Load Pre-processed PhreshPhish Data
    print("Loading Pre-processed PhreshPhish Data (40k)...")
    proc_dir = Path("data/processed/structural_xgb_40k")
    X = joblib.load(proc_dir / "X_phresh_40k.joblib")
    y = joblib.load(proc_dir / "y_phresh_40k.joblib")
    processor = joblib.load(proc_dir / "processor.joblib")
    
    # 2. Extract Feature Names and identify XPath features
    feature_names = processor.get_feature_names()
    
    # The last 25 features are the XPath TF-IDF features based on get_feature_names()
    xpath_indices = [i for i, name in enumerate(feature_names) if 'xpath_tfidf' in name]
    xpath_names = [feature_names[i].replace('xpath_tfidf_', '') for i in xpath_indices]
    
    print(f"Found {len(xpath_indices)} XPath structural features.")
    
    # Isolate only the XPath structural features
    X_xpath = X[:, xpath_indices]
    
    # 3. Visualization 1: PCA 2D Scatter Plot
    print("Computing PCA on DOM Structural Features...")
    pca = PCA(n_components=2, random_state=42)
    # Subsample for clearer visualization (5000 points)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_xpath), size=5000, replace=False)
    X_sample = X_xpath[sample_indices]
    y_sample = y[sample_indices]
    
    X_pca = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], 
        c=y_sample, cmap='coolwarm', alpha=0.6, s=15, edgecolors='none'
    )
    plt.title('DOM Tree Structure Space (PCA on XPath Sequences)', fontsize=16, pad=20)
    plt.xlabel('Principal Component 1 (DOM Complexity)', fontsize=12)
    plt.ylabel('Principal Component 2 (DOM Topology Variations)', fontsize=12)
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b4cc0', markersize=10, label='Benign Sites (Legit)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#b40426', markersize=10, label='Phishing Sites')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "dom_structure_pca.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Visualization 2: Top XPath Sequences by Mean TF-IDF
    print("Computing average XPath sequences for Phishing vs Legit...")
    # Unscale the data to get raw TF-IDF values if possible, but X is already scaled.
    # Because it is StandardScaled (mean=0), positive values mean "above average presence".
    
    mean_phish = np.mean(X_xpath[y == 1], axis=0)
    mean_benign = np.mean(X_xpath[y == 0], axis=0)
    
    # Difference (Phishing bias)
    diff = mean_phish - mean_benign
    
    # Sort by absolute difference
    sorted_indices = np.argsort(np.abs(diff))[::-1][:15] # Top 15 most discriminatory
    
    top_xpath_names = [xpath_names[i] for i in sorted_indices]
    top_diffs = diff[sorted_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    colors = ['#b40426' if val > 0 else '#3b4cc0' for val in top_diffs]
    
    bars = plt.barh(top_xpath_names, top_diffs, color=colors)
    plt.axvline(x=0, color='black', linewidth=1.5)
    
    plt.title('Most Discriminatory DOM Tree Structures (XPath Sequences)', fontsize=16, pad=20)
    plt.xlabel('Relative Presence (Standardized Z-Score Difference)', fontsize=12)
    plt.ylabel('HTML XPath Tag Sequence', fontsize=12)
    
    # Add text annotations to explain
    plt.text(0.5, 0.05, 'Stronger in\nPhishing', transform=plt.gca().transAxes, 
             fontsize=14, color='#b40426', alpha=0.7, weight='bold')
    plt.text(0.35, 0.05, 'Stronger in\nLegit', transform=plt.gca().transAxes, 
             fontsize=14, color='#3b4cc0', alpha=0.7, weight='bold')
             
    plt.gca().invert_yaxis() # Highest absolute difference at the top
    
    plt.savefig(out_dir / "dom_xpath_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations Complete. Plots saved to:")
    print("1. docs/assets/dom_structure_pca.png (2D Scatter showing structural separation)")
    print("2. docs/assets/dom_xpath_importance.png (Bar chart showing which exact DOM sequences are different)")

if __name__ == "__main__":
    main()

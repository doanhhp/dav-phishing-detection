import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Sampled Datasets ---")
    # Load ALL Old Data records
    df_old_url = pd.read_excel('data/raw/URL.xlsx')
    df_old_html = pd.read_excel('data/raw/html.xlsx')
    
    # Load PhreshPhish Data records
    df_phresh = pd.read_csv('data/raw/external/phreshphish.csv')

    print("--- 2. Extracting Structural Features ---")
    processor = StructuralProcessor({})
    
    # Process Old
    df_old_raw = df_old_url[['Data']].copy()
    df_old_raw['html'] = df_old_html['Data']
    X_old = processor.fit_transform(df_old_raw)
    
    # Process PhreshPhish
    # We only need 'Data' (URL) and 'html' columns for StructuralProcessor
    df_new_raw = df_phresh[['Data']].copy()
    df_new_raw['html'] = df_phresh['html']
    X_new = processor.transform(df_new_raw)

    print("--- 3. Performing t-SNE ---")
    # Concatenate the data because t-SNE must fit all points simultaneously
    X_combined = np.vstack((X_old, X_new))
    
    # Initialize t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, n_jobs=-1)
    
    print("Computing t-SNE embedding (this may take a minute)...")
    X_tsne = tsne.fit_transform(X_combined)
    
    # Split back into old and new
    X_old_tsne = X_tsne[:len(X_old)]
    X_new_tsne = X_tsne[len(X_old):]

    print("--- 4. Visualizing Domain Overlap ---")
    plt.figure(figsize=(12, 9))
    
    # Plot historical data
    plt.scatter(X_old_tsne[:, 0], X_old_tsne[:, 1], alpha=0.3, color='royalblue', label=f'Original Dataset ({len(X_old)} sites)', s=20)
    
    # Plot new PhreshPhish data
    plt.scatter(X_new_tsne[:, 0], X_new_tsne[:, 1], alpha=0.7, color='mediumseagreen', marker='s', label=f'PhreshPhish Dataset ({len(X_new)} sites)', s=30)
    
    plt.title('t-SNE Visualization: Original Dataset vs PhreshPhish Dataset', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Ensure artifacts directory exists
    out_dir = "C:/Users/Doanh1/.gemini/antigravity/brain/c9759465-bddd-49a6-afb6-1f85d35628bd/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tsne_phreshphish_overlap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE visualization to {out_path}")

if __name__ == "__main__":
    main()

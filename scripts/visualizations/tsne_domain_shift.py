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
    # Load 3000 Old Data records (t-SNE is O(N^2), so 3k + 1.8k = 4.8k is fast)
    df_old_url = pd.read_excel('data/raw/URL.xlsx').sample(n=3000, random_state=42)
    df_old_html = pd.read_excel('data/raw/html.xlsx').loc[df_old_url.index]
    
    # Load 1800 New Data records
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')

    print("--- 2. Extracting Structural Features ---")
    processor = StructuralProcessor({})
    
    # Process Old
    df_old_raw = df_old_url[['Data']].copy()
    df_old_raw['html'] = df_old_html['Data']
    X_old = processor.fit_transform(df_old_raw)
    
    # Process New
    df_new_raw = df_new_url[['Data']].copy()
    df_new_raw['html'] = df_new_html['Data']
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
    plt.figure(figsize=(14, 10))
    
    # Extract labels
    y_old = df_old_url['Category'].apply(lambda x: 'Benign' if 'ham' in str(x).lower() else 'Phishing').values
    y_new = df_new_url['Category'].apply(lambda x: 'Benign' if 'ham' in str(x).lower() else 'Phishing').values
    
    # Create masks
    mask_old_benign = y_old == 'Benign'
    mask_old_phishing = y_old == 'Phishing'
    mask_new_benign = y_new == 'Benign'
    mask_new_phishing = y_new == 'Phishing'
    
    # Plot historical data (2021)
    plt.scatter(X_old_tsne[mask_old_benign, 0], X_old_tsne[mask_old_benign, 1], alpha=0.4, color='royalblue', label='2021 Benign', s=25, marker='o')
    plt.scatter(X_old_tsne[mask_old_phishing, 0], X_old_tsne[mask_old_phishing, 1], alpha=0.4, color='darkorange', label='2021 Phishing', s=25, marker='o')
    
    # Plot new zero-day data (2026)
    plt.scatter(X_new_tsne[mask_new_benign, 0], X_new_tsne[mask_new_benign, 1], alpha=0.8, color='forestgreen', marker='^', label='2026 Benign (OOD)', s=40)
    plt.scatter(X_new_tsne[mask_new_phishing, 0], X_new_tsne[mask_new_phishing, 1], alpha=0.8, color='crimson', marker='^', label='2026 Phishing (OOD)', s=40)
    
    plt.title('t-SNE Visualization: Structural Evolution of Phishing vs Benign Domains', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(alpha=0.3)
    
    out_dir = "C:/Users/Doanh1/.gemini/antigravity/brain/4a4ead50-6758-44bf-8d9b-6d0f705fe101/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tsne_domain_overlap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE visualization to {out_path}")

if __name__ == "__main__":
    main()

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.features.structural import StructuralProcessor

def main():
    print("--- 1. Loading Full Datasets ---")
    
    # Load Pre-extracted 2021 Features
    X_train = joblib.load('data/processed/structural_xgb/X_train.joblib')
    X_test = joblib.load('data/processed/structural_xgb/X_test.joblib')
    X_old = np.vstack((X_train, X_test))
    
    y_train = joblib.load('data/processed/structural_xgb/y_train.joblib')
    y_test = joblib.load('data/processed/structural_xgb/y_test.joblib')
    y_old_num = np.concatenate((y_train, y_test))
    
    # Convert numerical labels to string labels
    y_old = np.where(y_old_num == 1, 'Phishing', 'Benign')

    # Load 2026 Data
    print("Loading 2026 data...")
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    y_new = df_new_url['Category'].apply(lambda x: 'Benign' if 'ham' in str(x).lower() else 'Phishing').values

    print("--- 2. Extracting Structural Features for 2026 Data ---")
    processor = joblib.load('data/processed/structural_xgb/processor.joblib')
    if not hasattr(processor, 'upper_bounds'):
        processor.upper_bounds = None
    df_new_raw = df_new_url[['Data']].copy()
    df_new_raw['html'] = df_new_html['Data']
    
    X_new = processor.transform(df_new_raw)
    if hasattr(X_new, 'toarray'):
        X_new = X_new.toarray()
        
    if hasattr(X_old, 'toarray'):
        X_old = X_old.toarray()

    print("--- 3. Performing PCA and t-SNE on 50,000+ points ---")
    X_combined = np.vstack((X_old, X_new))
    
    # Use PCA to reduce to 50 dims first (standard practice for speeding up massive t-SNE)
    print("Running PCA reduction to 50 dimensions...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_combined)

    print("Computing t-SNE embedding (this may take a few minutes)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_pca)
    
    X_old_tsne = X_tsne[:len(X_old)]
    X_new_tsne = X_tsne[len(X_old):]

    print("--- 4. Visualizing Domain Overlap ---")
    plt.figure(figsize=(14, 10))
    
    mask_old_benign = y_old == 'Benign'
    mask_old_phishing = y_old == 'Phishing'
    mask_new_benign = y_new == 'Benign'
    mask_new_phishing = y_new == 'Phishing'
    
    plt.scatter(X_old_tsne[mask_old_benign, 0], X_old_tsne[mask_old_benign, 1], alpha=0.3, color='royalblue', label=f'2021 Benign ({sum(mask_old_benign):,})', s=15, marker='o')
    plt.scatter(X_old_tsne[mask_old_phishing, 0], X_old_tsne[mask_old_phishing, 1], alpha=0.3, color='darkorange', label=f'2021 Phishing ({sum(mask_old_phishing):,})', s=15, marker='o')
    
    plt.scatter(X_new_tsne[mask_new_benign, 0], X_new_tsne[mask_new_benign, 1], alpha=0.8, color='forestgreen', marker='^', label=f'2026 Benign ({sum(mask_new_benign):,})', s=30)
    plt.scatter(X_new_tsne[mask_new_phishing, 0], X_new_tsne[mask_new_phishing, 1], alpha=0.8, color='crimson', marker='^', label=f'2026 Phishing ({sum(mask_new_phishing):,})', s=30)
    
    plt.title('t-SNE visualization', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(alpha=0.3)
    
    out_dir = "C:/Users/Doanh1/.gemini/antigravity/brain/4a4ead50-6758-44bf-8d9b-6d0f705fe101/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tsne_full_domain_overlap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig('d:/Desktop/PhishingDetection/tsne_full_domain_overlap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Full t-SNE visualization to {out_path}")

if __name__ == "__main__":
    main()

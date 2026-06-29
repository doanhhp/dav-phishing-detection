import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / 'data' / 'processed' / 'standardized'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Main Dataset (2021)
    print("Processing Main Dataset...")
    df_main_url = pd.read_excel(project_root / 'data' / 'raw' / 'URL.xlsx')
    df_main_html = pd.read_excel(project_root / 'data' / 'raw' / 'html.xlsx')
    
    df_main = df_main_url[['Data', 'Category']].copy()
    df_main['html'] = df_main_html['Data'].values
    # Map spam/ham to 1/0
    df_main['Category'] = df_main['Category'].map({'spam': 1, 'ham': 0}).astype(int)
    
    out_main = output_dir / 'main_dataset.parquet'
    df_main.to_parquet(out_main, index=False)
    print(f"Saved Main Dataset to {out_main} ({len(df_main)} rows)")
    
    # 2. OOD Dataset (2026)
    print("Processing OOD Dataset...")
    df_ood_url = pd.read_excel(project_root / 'data' / 'raw' / 'OOD_URL.xlsx')
    df_ood_html = pd.read_excel(project_root / 'data' / 'raw' / 'OOD_html.xlsx')
    
    df_ood = df_ood_url[['Data', 'Category']].copy()
    df_ood['html'] = df_ood_html['Data'].values
    
    # Map categories to 1/0
    phish_cats = ['phishing', 'malware', 'spam']
    benign_cats = ['benign', 'ham']
    
    def map_ood(cat):
        if pd.isna(cat): return 0
        cat_lower = str(cat).lower()
        if cat_lower in phish_cats: return 1
        if cat_lower in benign_cats: return 0
        return 0
        
    df_ood['Category'] = df_ood['Category'].apply(map_ood).astype(int)
    
    out_ood = output_dir / 'ood_dataset.parquet'
    df_ood.to_parquet(out_ood, index=False)
    print(f"Saved OOD Dataset to {out_ood} ({len(df_ood)} rows)")
    
    # 3. PhreshPhish Dataset
    print("Processing PhreshPhish Dataset...")
    df_phresh = pd.read_parquet(project_root / 'data' / 'raw' / 'external' / 'phreshphish_40k.parquet')
    
    # PhreshPhish already has Data, html, Category. Just enforce types.
    df_phresh['Category'] = df_phresh['Category'].astype(int)
    
    # Ensure only the 3 required columns
    df_phresh = df_phresh[['Data', 'html', 'Category']]
    
    out_phresh = output_dir / 'phreshphish_dataset.parquet'
    df_phresh.to_parquet(out_phresh, index=False)
    print(f"Saved PhreshPhish Dataset to {out_phresh} ({len(df_phresh)} rows)")

if __name__ == "__main__":
    main()

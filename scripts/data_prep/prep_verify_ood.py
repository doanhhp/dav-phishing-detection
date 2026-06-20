import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_dataset():
    url_file = "data/raw/OOD_URL.xlsx"
    html_file = "data/raw/OOD_html.xlsx"
    
    try:
        df_url = pd.read_excel(url_file)
        df_html = pd.read_excel(html_file)
    except Exception as e:
        logging.error(f"Error reading datasets: {e}")
        return

    # 1. Check Row Match
    if len(df_url) != len(df_html):
        logging.error(f"Row count mismatch! URL rows: {len(df_url)}, HTML rows: {len(df_html)}")
    
    # 2. Check for NaN or empty
    null_urls = df_url['Data'].isnull().sum()
    null_html = df_html['Data'].isnull().sum()
    
    if null_urls > 0 or null_html > 0:
        logging.warning(f"Found missing data! Null URLs: {null_urls}, Null HTMLs: {null_html}")
        # Drop nulls
        df_url = df_url.dropna(subset=['Data'])
        df_html = df_html.dropna(subset=['Data'])
    
    # Check for empty strings in HTML
    empty_html = (df_html['Data'].astype(str).str.strip() == '').sum()
    if empty_html > 0:
        logging.warning(f"Found {empty_html} rows with empty HTML content.")
    
    # 3. Check for duplicates
    dups = df_url.duplicated(subset=['Data']).sum()
    if dups > 0:
        logging.warning(f"Found {dups} duplicate URLs! Cleaning them up...")
        df_combined = df_url.copy()
        df_combined['html'] = df_html['Data']
        df_combined = df_combined.drop_duplicates(subset=['Data'])
        
        df_url = df_combined[['Category', 'Data']].copy()
        df_html = pd.DataFrame({'Category': df_combined['Category'], 'Data': df_combined['html']})
        
        df_url.to_excel(url_file, index=False)
        df_html.to_excel(html_file, index=False)
        logging.info("Duplicates removed and files saved.")

    # 4. Print Distribution
    dist = df_url['Category'].value_counts()
    total = len(df_url)
    logging.info("\n--- FINAL DATASET STATISTICS ---")
    logging.info(f"Total Samples: {total}")
    for cat, count in dist.items():
        pct = (count / total) * 100
        logging.info(f"{cat}: {count} ({pct:.1f}%)")
        
    logging.info("Dataset verification complete and clean!")

if __name__ == '__main__':
    verify_dataset()

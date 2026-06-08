import os
import io
import json
import zipfile
import requests
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_legit_urls(limit: int, existing_urls: set) -> list:
    """Fetch legitimate URLs from the Tranco Top 1M list, skipping existing."""
    logger.info("Downloading Tranco Top 1M list for legitimate URLs...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, names=['rank', 'domain'])
                
        urls = []
        for d in df['domain'].tolist():
            formatted_url = "https://" + str(d)
            if formatted_url not in existing_urls:
                urls.append(formatted_url)
            if len(urls) >= limit:
                break
                
        logger.info(f"Successfully loaded {len(urls)} *new* legitimate URLs from Tranco.")
        return urls
    except Exception as e:
        logger.error(f"Failed to fetch Tranco list: {e}")
        return []

def get_phishing_urls(limit: int, existing_urls: set, phishtank_key: str = None) -> list:
    """Fetch phishing URLs from PhishTank or fallback to OpenPhish, skipping existing."""
    urls = []
    
    # Try PhishTank first
    logger.info("Attempting to fetch phishing URLs from PhishTank...")
    pt_url = "http://data.phishtank.com/data/online-valid.json"
    if phishtank_key:
        pt_url += f"?app_key={phishtank_key}"
        
    try:
        response = requests.get(pt_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        for item in data:
            if item.get('online') == 'yes' and item.get('verified') == 'yes':
                url = item.get('url')
                if url not in existing_urls:
                    urls.append(url)
                if len(urls) >= limit:
                    break
        if urls:
            logger.info(f"Successfully loaded {len(urls)} *new* phishing URLs from PhishTank.")
            return urls
    except Exception as e:
        logger.warning(f"Failed to fetch from PhishTank: {e}")
        logger.info("Falling back to OpenPhish free feed...")
        
    # Fallback to OpenPhish
    try:
        op_url = "https://openphish.com/feed.txt"
        response = requests.get(op_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        
        for line in lines:
            url = line.strip()
            if url and url not in existing_urls:
                urls.append(url)
            if len(urls) >= limit:
                break
                
        logger.info(f"Successfully loaded {len(urls)} *new* phishing URLs from OpenPhish.")
        return urls
    except Exception as e:
        logger.error(f"Failed to fetch from OpenPhish: {e}")
        return []

def fetch_html(url: str, timeout: int = 5) -> str:
    """Fetch HTML content for a single URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        # Check if content is HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return None
            
        return response.text
    except Exception:
        # Silently fail for timeout or connection errors
        return None

def crawl_dataset(num_legit: int, num_phish: int, output_dir: str, phishtank_key: str = None, timeout: int = 5, append: bool = False):
    """Crawl URLs and their HTML, saving successfully matched pairs to excel files."""
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    url_file = out_path / "OOD_URL.xlsx"
    html_file = out_path / "OOD_html.xlsx"
    
    existing_urls = set()
    df_existing_url = pd.DataFrame()
    df_existing_html = pd.DataFrame()
    
    if append and url_file.exists() and html_file.exists():
        logger.info("Loading existing dataset for appending and deduplication...")
        try:
            df_existing_url = pd.read_excel(url_file, engine='calamine')
            df_existing_html = pd.read_excel(html_file, engine='calamine')
        except:
            df_existing_url = pd.read_excel(url_file)
            df_existing_html = pd.read_excel(html_file)
            
        existing_urls = set(df_existing_url['Data'].tolist())
        logger.info(f"Found {len(existing_urls)} existing URLs.")

    legit_urls = get_legit_urls(num_legit, existing_urls)
    phishing_urls = get_phishing_urls(num_phish, existing_urls, phishtank_key)
    
    # Combine and label
    targets = [('ham', url) for url in legit_urls] + [('spam', url) for url in phishing_urls]
    
    if not targets:
        logger.info("No new URLs to fetch. Exiting.")
        return

    logger.info(f"Starting HTML extraction for {len(targets)} total NEW URLs with a {timeout}s timeout...")
    
    successful_urls = []
    successful_html = []
    
    import re
    # ILLEGAL_CHARACTERS_RE pattern for Excel
    illegal_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    
    for category, url in tqdm(targets, desc="Fetching HTML"):
        html = fetch_html(url, timeout=timeout)
        if html and len(html.strip()) > 100: # Ensure valid HTML size
            clean_html = illegal_chars.sub('', html)
            successful_urls.append({'Category': category, 'Data': url})
            successful_html.append({'Category': category, 'Data': clean_html})
            
    logger.info(f"Successfully fetched {len(successful_urls)}/{len(targets)} new websites.")
    
    df_new_url = pd.DataFrame(successful_urls)
    df_new_html = pd.DataFrame(successful_html)
    
    if append and not df_existing_url.empty:
        df_url = pd.concat([df_existing_url, df_new_url], ignore_index=True)
        df_html = pd.concat([df_existing_html, df_new_html], ignore_index=True)
    else:
        df_url = df_new_url
        df_html = df_new_html
        
    # Final Deduplication and Cleaning Check
    logger.info("Running final deduplication check...")
    df_combined = df_url.copy()
    df_combined['html'] = df_html['Data']
    df_combined = df_combined.drop_duplicates(subset=['Data']) # drop duplicate URLs
    
    df_final_url = df_combined[['Category', 'Data']].copy()
    df_final_html = pd.DataFrame({'Category': df_combined['Category'], 'Data': df_combined['html']})
    
    logger.info(f"Final dataset size: {len(df_final_url)} total samples.")
    
    logger.info(f"Saving {url_file}...")
    df_final_url.to_excel(url_file, index=False)
    
    logger.info(f"Saving {html_file}...")
    df_final_html.to_excel(html_file, index=False)
    
    logger.info("Data crawling and export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Web Crawler for Out-Of-Distribution Phishing Testing")
    parser.add_argument("--legit", type=int, default=50, help="Number of legitimate URLs to attempt to fetch")
    parser.add_argument("--phish", type=int, default=50, help="Number of phishing URLs to attempt to fetch")
    parser.add_argument("--timeout", type=int, default=5, help="Seconds to wait before skipping a website")
    parser.add_argument("--phishtank_key", type=str, default=None, help="Optional API key for PhishTank")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset and avoid duplicates")
    
    args = parser.parse_args()
    crawl_dataset(args.legit, args.phish, args.output, args.phishtank_key, args.timeout, args.append)

"""Script to export structural features for both training and OOD datasets."""

import os
import pandas as pd
import logging
from tqdm import tqdm
import sys

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.structural import StructuralProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_data():
    processor = StructuralProcessor({})
    feature_names = processor.get_feature_names()
    
    # 1. Process Training Data
    logger.info("Loading original training data...")
    url_df = pd.read_excel('data/raw/URL.xlsx')
    html_df = pd.read_excel('data/raw/html.xlsx')
    
    # Merge and align
    if 'Label' in url_df.columns:
        url_df = url_df.rename(columns={'Label': 'Category'})
    
    urls = url_df['Data'].values
    htmls = html_df['Data'].values
    labels = url_df['Category'].values
    
    logger.info("Extracting structural features for training data (this may take a minute)...")
    train_features = []
    for u, h in tqdm(zip(urls, htmls), total=len(urls)):
        train_features.append(processor._extract_features(u, h))
        
    df_train = pd.DataFrame(train_features, columns=feature_names)
    df_train['Label'] = labels
    
    os.makedirs('data/processed', exist_ok=True)
    train_out = 'data/processed/structural_data_train.csv'
    df_train.to_csv(train_out, index=False)
    logger.info(f"Saved {len(df_train)} rows to {train_out}")
    
    # 2. Process OOD Data
    logger.info("Loading OOD data...")
    try:
        ood_url_df = pd.read_excel('data/raw/OOD_URL.xlsx')
        ood_html_df = pd.read_excel('data/raw/OOD_html.xlsx')
        
        ood_urls = ood_url_df['Data'].values
        ood_htmls = ood_html_df['Data'].values
        ood_labels = ood_url_df['Category'].values
        
        logger.info("Extracting structural features for OOD data...")
        ood_features = []
        for u, h in tqdm(zip(ood_urls, ood_htmls), total=len(ood_urls)):
            ood_features.append(processor._extract_features(u, h))
            
        df_ood = pd.DataFrame(ood_features, columns=feature_names)
        df_ood['Label'] = ood_labels
        
        ood_out = 'data/processed/structural_data_ood.csv'
        df_ood.to_csv(ood_out, index=False)
        logger.info(f"Saved {len(df_ood)} rows to {ood_out}")
        
    except FileNotFoundError:
        logger.warning("OOD data files not found. Skipping OOD export.")

if __name__ == "__main__":
    export_data()

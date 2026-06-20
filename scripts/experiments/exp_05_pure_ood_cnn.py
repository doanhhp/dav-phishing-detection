import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score

from src.features.multimodal import MultimodalProcessor
from src.models.webphish_cnn import WebPhish_CNN

# Optimize TensorFlow to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    print("--- 1. Loading New Zero-Day Dataset (OOD) ---")
    df_ood_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_ood_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    if 'Label' in df_ood_url.columns:
        df_ood_url = df_ood_url.rename(columns={'Label': 'Category'})
    
    y = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    # MultimodalProcessor expects 'url' and 'html' columns
    df_raw = pd.DataFrame()
    df_raw['url'] = df_ood_url['Data'].astype(str)
    df_raw['html'] = df_ood_html['Data'].astype(str)
    
    print("--- 2. Performing 80/20 Train/Test Split ---")
    df_train, df_test, y_train, y_test = train_test_split(
        df_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 3. Extracting Multimodal Text Features (URL Char & HTML Words) ---")
    processor = MultimodalProcessor(config={
        "url_max_length": 180,
        "html_max_length": 2000,
        "html_vocab_size": 20000 # Reduced slightly for speed on CPU
    })
    
    print("Fitting processor on training data...")
    X_train = processor.fit_transform(df_train)
    print("Transforming testing data...")
    X_test = processor.transform(df_test)

    print(f"Training set: {len(df_train)} samples")
    print(f"Testing set: {len(df_test)} samples")

    print("\n--- 4. Building & Training WebPhish CNN ---")
    # Using 10 epochs. The original paper used early stopping.
    config = {
        "epochs": 10,
        "batch_size": 64,
        "url_max_length": 180,
        "html_max_length": 2000,
        "url_vocab_size": 130,
        "html_vocab_size": 20000,
        "embedding_dim": 16,
        "filters": 32,
        "kernel_size": 8,
        "dropout": 0.2
    }
    
    cnn = WebPhish_CNN(config=config)
    cnn.fit(X_train, y_train)

    print("\n--- 5. Evaluating WebPhish CNN on New Test Set ---")
    preds = cnn.predict(X_test)
    probs = cnn.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    
    print(f"WebPhish CNN (In-Domain 2026) -> Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

    print("\nDetailed Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()

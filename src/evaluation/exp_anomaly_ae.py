"""Experiment: Anomaly Detection Autoencoder for Zero-Day Phishing."""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from src.models.anomaly_ae import AnomalyAutoencoder

def main():
    print("--- 1. Loading Training Data & Processor ---")
    struct_X_train = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\structural_rf\X_train.joblib")
    struct_y_train = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\structural_rf\y_train.joblib")
    
    struct_proc_path = r"D:\Desktop\PhishingDetection\data\processed\structural_rf\processor.joblib"
    struct_proc = joblib.load(struct_proc_path)
    
    # Filter ONLY 'ham' (legitimate) data for training
    X_train_ham = struct_X_train[struct_y_train == 0]
    print(f"Training Autoencoder on {X_train_ham.shape[0]} legitimate samples...")

    print("--- 2. Building & Training Autoencoder ---")
    input_dim = X_train_ham.shape[1]
    ae = AnomalyAutoencoder(input_dim=input_dim, config={'encoding_dim': 8, 'hidden_dim': 24, 'epochs': 20})
    
    # Train the AE
    ae.fit(X_train_ham, epochs=15, batch_size=128, validation_split=0.1)

    print("--- 3. Evaluating on Zero-Shot OOD Dataset ---")
    ood_url_path = r"D:\Desktop\PhishingDetection\data\raw\OOD_URL.xlsx"
    ood_html_path = r"D:\Desktop\PhishingDetection\data\raw\OOD_html.xlsx"
    
    df_ood_url = pd.read_excel(ood_url_path)
    df_ood_html = pd.read_excel(ood_html_path)
    
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    print("Extracting OOD Structural Features...")
    df_struct_raw = df_ood_url[['Data']].copy()
    df_struct_raw['html'] = df_ood_html['Data']
    struct_X_ood = struct_proc.transform(df_struct_raw)

    print("Calculating Reconstruction Errors...")
    ood_mse = ae.get_reconstruction_error(struct_X_ood)
    
    # Calculate ROC AUC
    auc = roc_auc_score(y_ood, ood_mse)
    print(f"\nZero-Shot OOD Anomaly Detection AUC: {auc:.4f}")
    
    # Plot MSE distributions
    plt.figure(figsize=(10, 6))
    plt.hist(ood_mse[y_ood == 0], bins=50, alpha=0.5, color='blue', label='Legitimate (Ham) MSE')
    plt.hist(ood_mse[y_ood == 1], bins=50, alpha=0.5, color='red', label='Phishing (Spam) MSE')
    plt.axvline(np.percentile(ood_mse[y_ood == 0], 90), color='k', linestyle='dashed', linewidth=2, label='90th Percentile Ham Threshold')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Anomaly Detection: Reconstruction Error on Zero-Day Phishing')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"D:\Desktop\PhishingDetection\doc\assets\anomaly_ae_mse.png")
    print("Saved MSE distribution plot to doc/assets/anomaly_ae_mse.png")
    
    # Set threshold based on 90th percentile of HAM (accepting 10% false positive rate)
    threshold = np.percentile(ood_mse[y_ood == 0], 90)
    ood_preds = (ood_mse > threshold).astype(int)
    
    acc = accuracy_score(y_ood, ood_preds)
    print(f"\nFINAL ANOMALY AE ZERO-SHOT OOD ACCURACY (Threshold={threshold:.4f}): {acc:.4f}")
    print(classification_report(y_ood, ood_preds))

if __name__ == "__main__":
    main()

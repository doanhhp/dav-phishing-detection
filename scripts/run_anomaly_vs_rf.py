import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, f1_score
from src.models.anomaly_ae import AnomalyAutoencoder
from src.models.structural_rf import Structural_RF

def main():
    print("--- 1. Loading Original Training Data (Historical) ---")
    # Load 5k of original training data
    df_train_url = pd.read_excel('data/raw/URL.xlsx').head(5000)
    df_train_html = pd.read_excel('data/raw/html.xlsx').head(5000)
    
    if 'Label' in df_train_url.columns:
        df_train_url = df_train_url.rename(columns={'Label': 'Category'})
    
    y_train = df_train_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    print("Extracting Structural Features for Training...")
    # Load processor from disk if exists or train new
    proc_path = "data/processed/structural_rf/processor.joblib"
    if os.path.exists(proc_path):
        struct_proc = joblib.load(proc_path)
        # transform the 5k
        df_train_raw = df_train_url[['Data']].copy()
        df_train_raw['html'] = df_train_html['Data']
        X_train = struct_proc.transform(df_train_raw)
    else:
        from src.features.structural import StructuralProcessor
        struct_proc = StructuralProcessor({})
        df_train_raw = df_train_url[['Data']].copy()
        df_train_raw['html'] = df_train_html['Data']
        X_train = struct_proc.fit_transform(df_train_raw)

    # Filter ONLY 'ham' (legitimate) data for Autoencoder training
    X_train_ham = X_train[y_train == 0]
    print(f"Training Autoencoder exclusively on {X_train_ham.shape[0]} historical legitimate samples...")

    print("\n--- 2. Building & Training Models ---")
    input_dim = X_train.shape[1]
    
    # Train Autoencoder
    ae = AnomalyAutoencoder(input_dim=input_dim, config={'encoding_dim': 16, 'hidden_dim': 32, 'epochs': 25, 'learning_rate': 0.002})
    ae.fit(X_train_ham, epochs=25, batch_size=128, validation_split=0.1)

    # Train Random Forest
    print(f"\nTraining Structural RF on all {X_train.shape[0]} historical samples (ham + spam)...")
    rf = Structural_RF(config={})
    rf.fit(X_train, y_train)

    print("\n--- 3. Evaluating on Zero-Shot OOD Dataset (1.8k New Samples) ---")
    df_ood_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_ood_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    print("Extracting OOD Structural Features...")
    df_ood_raw = df_ood_url[['Data']].copy()
    df_ood_raw['html'] = df_ood_html['Data']
    X_ood = struct_proc.transform(df_ood_raw)

    print("\n--- 4. Calculating Predictions ---")
    # Random Forest Predictions
    rf_preds = rf.predict(X_ood)
    rf_probs = rf.predict_proba(X_ood)[:, 1]
    
    # Autoencoder Predictions
    ood_mse = ae.get_reconstruction_error(X_ood)
    
    # Set threshold based on 90th percentile of HAM from the *OOD* dataset (adaptive) or *Train* dataset (strict)
    # Strict resilience test: using threshold learned from training ham
    train_mse = ae.get_reconstruction_error(X_train_ham)
    threshold = np.percentile(train_mse, 90)
    ae_preds = (ood_mse > threshold).astype(int)

    print("\n--- 5. Results & Comparisons ---")
    rf_acc = accuracy_score(y_ood, rf_preds)
    rf_auc = roc_auc_score(y_ood, rf_probs)
    rf_f1 = f1_score(y_ood, rf_preds)
    
    ae_acc = accuracy_score(y_ood, ae_preds)
    ae_auc = roc_auc_score(y_ood, ood_mse)
    ae_f1 = f1_score(y_ood, ae_preds)
    
    print(f"Random Forest (Historical Data) -> Acc: {rf_acc:.4f} | AUC: {rf_auc:.4f} | F1: {rf_f1:.4f}")
    print(f"Anomaly AE  (Historical Data) -> Acc: {ae_acc:.4f} | AUC: {ae_auc:.4f} | F1: {ae_f1:.4f}")
    
    # Plotting
    os.makedirs('docs/assets/model_comparisons', exist_ok=True)
    
    # ROC Curve Comparison
    plt.figure(figsize=(8, 6))
    rf_fpr, rf_tpr, _ = roc_curve(y_ood, rf_probs)
    ae_fpr, ae_tpr, _ = roc_curve(y_ood, ood_mse)
    
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', linewidth=2)
    plt.plot(ae_fpr, ae_tpr, label=f'Anomaly AE (AUC = {ae_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Zero-Day Phishing Detection: Anomaly AE vs Random Forest')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/assets/model_comparisons/anomaly_vs_rf_roc.png')
    
    # MSE Distribution Plot for AE
    plt.figure(figsize=(10, 6))
    plt.hist(ood_mse[y_ood == 0], bins=50, alpha=0.5, color='blue', label='Legitimate (Ham) MSE')
    plt.hist(ood_mse[y_ood == 1], bins=50, alpha=0.5, color='red', label='Phishing (Spam) MSE')
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Autoencoder Anomaly Detection on Zero-Day Data')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/assets/model_comparisons/anomaly_mse_dist.png')
    
    print("\nVisualizations saved to docs/assets/model_comparisons/")
    print("\nDetailed Reports:")
    print("Random Forest:")
    print(classification_report(y_ood, rf_preds))
    print("Anomaly AE:")
    print(classification_report(y_ood, ae_preds))

if __name__ == "__main__":
    main()

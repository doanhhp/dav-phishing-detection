"""Experiment: Deep Learning Few-Shot Transfer Learning."""

import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(r"D:\Desktop\PhishingDetection")

def build_dnn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,), name='dense_1'),
        Dropout(0.2, name='drop_1'),
        Dense(32, activation='relu', name='dense_2'),
        Dropout(0.2, name='drop_2'),
        Dense(16, activation='relu', name='dense_3'),
        Dense(1, activation='sigmoid', name='output')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("--- 1. Loading Historical Training Data (For DNN Pre-training) ---")
    X_train_hist = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\structural_rf\X_train.joblib")
    y_train_hist = joblib.load(r"D:\Desktop\PhishingDetection\data\processed\structural_rf\y_train.joblib")
    input_dim = X_train_hist.shape[1]

    print("--- 2. Pre-Training Base DNN Model (Historical Data) ---")
    base_dnn = build_dnn(input_dim)
    base_dnn.fit(X_train_hist, y_train_hist, epochs=3, batch_size=256, verbose=0, validation_split=0.1)

    print("--- 3. Loading OOD Data ---")
    ood_url_path = project_root / "data/raw/OOD_URL.xlsx"
    ood_html_path = project_root / "data/raw/OOD_html.xlsx"
    df_ood_url = pd.read_excel(ood_url_path)
    df_ood_html = pd.read_excel(ood_html_path)
    df_ood_url['html'] = df_ood_html['Data']
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    processor = joblib.load(project_root / "data/processed/structural_rf/processor.joblib")
    X_ood = processor.transform(df_ood_url[['Data', 'html']])

    # Zero-shot evaluation
    zero_shot_preds = (base_dnn.predict(X_ood, verbose=0) > 0.5).astype(int)
    print(f"Zero-Shot DNN OOD Accuracy: {accuracy_score(y_ood, zero_shot_preds):.4f}")

    sample_sizes = [10, 20, 50, 100, 200]
    n_seeds = 3
    results = []

    print("--- 4. Running Deep Learning Few-Shot Comparisons ---")
    for n_samples in sample_sizes:
        print(f"Evaluating with {n_samples} new zero-day samples...")
        
        acc_scratch_dnn = []
        acc_transfer_dnn = []
        acc_scratch_rf = []
        
        for seed in range(n_seeds):
            X_train_few, X_test_few, y_train_few, y_test_few = train_test_split(
                X_ood, y_ood, train_size=n_samples, random_state=seed, stratify=y_ood
            )
            
            # --- Method A: RF Scratch (Our Champion) ---
            m_rf = RandomForestClassifier(n_estimators=50, random_state=seed)
            m_rf.fit(X_train_few, y_train_few)
            acc_scratch_rf.append(accuracy_score(y_test_few, m_rf.predict(X_test_few)))
            
            # --- Method B: DNN from Scratch ---
            m_dnn_scratch = build_dnn(input_dim)
            m_dnn_scratch.fit(X_train_few, y_train_few, epochs=10, batch_size=4, verbose=0)
            preds_dnn = (m_dnn_scratch.predict(X_test_few, verbose=0) > 0.5).astype(int)
            acc_scratch_dnn.append(accuracy_score(y_test_few, preds_dnn))
            
            # --- Method C: DNN Transfer Learning (Fine-Tuning) ---
            # Clone base weights
            m_dnn_transfer = build_dnn(input_dim)
            m_dnn_transfer.set_weights(base_dnn.get_weights())
            
            # Freeze all layers except the last Dense layer
            for layer in m_dnn_transfer.layers[:-1]:
                layer.trainable = False
                
            # Recompile to apply frozen layers
            m_dnn_transfer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                                   loss='binary_crossentropy', metrics=['accuracy'])
            
            # Fine-tune the output layer
            m_dnn_transfer.fit(X_train_few, y_train_few, epochs=10, batch_size=4, verbose=0)
            preds_transfer = (m_dnn_transfer.predict(X_test_few, verbose=0) > 0.5).astype(int)
            acc_transfer_dnn.append(accuracy_score(y_test_few, preds_transfer))

        results.append({'N_Samples': n_samples, 'Model': 'RF_Scratch', 'Accuracy': np.mean(acc_scratch_rf)})
        results.append({'N_Samples': n_samples, 'Model': 'DNN_Scratch', 'Accuracy': np.mean(acc_scratch_dnn)})
        results.append({'N_Samples': n_samples, 'Model': 'DNN_Transfer', 'Accuracy': np.mean(acc_transfer_dnn)})

    df_res = pd.DataFrame(results)
    
    # Save Tabular Results
    report_path = project_root / "docs/few_shot_report.md"
    
    # We will append to the existing report to show the DNN comparison
    pivot_df = df_res.pivot(index='N_Samples', columns='Model', values='Accuracy')
    
    print("\n--- DEEP LEARNING FEW-SHOT RESULTS ---")
    print(pivot_df)
    
    with open(report_path, 'a') as f:
        f.write("\n## Deep Learning Transfer Learning Results\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n\n*DNN Transfer Freezes the base layers trained on historical data and fine-tunes only the final dense layer.*\n")
        
    print(f"Saved tabular report to {report_path}")

if __name__ == "__main__":
    main()

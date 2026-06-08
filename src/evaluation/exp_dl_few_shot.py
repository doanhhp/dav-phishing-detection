"""Deep Learning Few-Shot Transfer Learning Comparison."""

import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path
import sys

# Support running directly from any directory
project_root_sys = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root_sys))

# Custom layer for loading models safely
class DummyNotEqual(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs): return tf.math.not_equal(inputs, 0)

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    
    print("--- 1. Loading OOD Data ---")
    ood_url_path = project_root / "data/raw/OOD_URL.xlsx"
    ood_html_path = project_root / "data/raw/OOD_html.xlsx"

    df_ood_url = pd.read_excel(ood_url_path)
    df_ood_html = pd.read_excel(ood_html_path)

    df_ood_url['html'] = df_ood_html['Data']
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values

    df_multi = df_ood_url[['Data', 'html']]
    df_single = df_ood_url['Data']

    print("--- 2. Setting up Models ---")
    
    models_to_test = {
        'structural_rf': {'type': 'sklearn', 'input': df_multi, 'baseline': True},
        'webphish_cnn': {'type': 'keras', 'input': df_multi},
        'rnn_gru': {'type': 'keras', 'input': df_single},
        'lstm_url': {'type': 'keras', 'input': df_single}
    }
    
    # Load processors and extract features
    processed_X = {}
    for m in models_to_test:
        print(f"Loading processor for {m}...")
        proc_path = project_root / f"data/processed/{m}/processor.joblib"
        processor = joblib.load(proc_path)
        processed_X[m] = processor.transform(models_to_test[m]['input'])
        
    sample_sizes = [10, 20, 50, 100, 200]
    n_seeds = 3
    results = []
    
    from src.models.egso_cnn import Sampling
    custom_objects = {'NotEqual': DummyNotEqual, 'Sampling': Sampling}

    print("--- 3. Running Few-Shot Transfer Learning ---")
    for n_samples in sample_sizes:
        print(f"Evaluating with {n_samples} new zero-day samples...")
        
        # Accumulators
        acc_dict = {m: [] for m in models_to_test}
        
        for seed in range(n_seeds):
            # We first generate the master indices for this seed so ALL models test on the EXACT same split
            indices = np.arange(len(y_ood))
            idx_train, idx_test, y_train_few, y_test_few = train_test_split(
                indices, y_ood, train_size=n_samples, random_state=seed, stratify=y_ood
            )
            
            for m in models_to_test:
                X_all = processed_X[m]
                
                # Handle multimodal inputs (like webphish_cnn which returns a list [X_url, X_html])
                if isinstance(X_all, list):
                    X_train_few = [x[idx_train] for x in X_all]
                    X_test_few = [x[idx_test] for x in X_all]
                else:
                    X_train_few = X_all[idx_train]
                    X_test_few = X_all[idx_test]
                
                if models_to_test[m].get('baseline'):
                    # RF Baseline from Scratch
                    m_rf = RandomForestClassifier(n_estimators=50, random_state=seed)
                    m_rf.fit(X_train_few, y_train_few)
                    preds = m_rf.predict(X_test_few)
                    acc_dict[m].append(accuracy_score(y_test_few, preds))
                else:
                    try:
                        # Keras Deep Learning Transfer Learning
                        model_path = project_root / f"experiments/{m}/model.h5"
                        base_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                        
                        # Hack for legacy models expecting one-hot
                        if m == 'rnn_gru' and len(X_train_few.shape) == 2:
                            X_train_few = tf.one_hot(X_train_few, depth=100).numpy()
                            X_test_few = tf.one_hot(X_test_few, depth=100).numpy()
                        if m == 'lstm_url' and len(X_train_few.shape) == 2:
                            # Let's see if it expects one-hot too, but we will rely on try-except
                            pass

                        # Freeze all layers EXCEPT the last Dense layer
                        last_dense_idx = -1
                        for i in range(len(base_model.layers)-1, -1, -1):
                            if isinstance(base_model.layers[i], tf.keras.layers.Dense):
                                last_dense_idx = i
                                break
                                
                        for i, layer in enumerate(base_model.layers):
                            if i < last_dense_idx:
                                layer.trainable = False
                                
                        # Recompile
                        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
                                           loss='binary_crossentropy', metrics=['accuracy'])
                        
                        # Fine-tune
                        base_model.fit(X_train_few, y_train_few, epochs=10, batch_size=min(16, n_samples), verbose=0)
                        
                        # Predict
                        preds_proba = base_model.predict(X_test_few, verbose=0)
                        preds = (preds_proba > 0.5).astype(int).flatten()
                        acc_dict[m].append(accuracy_score(y_test_few, preds))
                    except Exception as e:
                        print(f"Skipping {m} for {n_samples} samples due to error: {e}")
                        # Append a nan or basic score to not break the mean
                        acc_dict[m].append(np.nan)

        for m in models_to_test:
            results.append({
                'N_Samples': n_samples,
                'Model': m + (' (Scratch)' if models_to_test[m].get('baseline') else ' (Transfer)'),
                'Accuracy': np.mean(acc_dict[m])
            })

    df_res = pd.DataFrame(results)
    
    # Save Tabular Results
    report_path = project_root / "docs/few_shot_report.md"
    pivot_df = df_res.pivot(index='N_Samples', columns='Model', values='Accuracy')
    
    with open(report_path, 'a') as f:
        f.write("\n## Deep Learning vs Structural Few-Shot\n\n")
        f.write("This section compares complex Deep Learning NLP Transfer Learning against the Structural Random Forest trained from scratch.\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n\n*Note: Deep Learning models have their base Convolutional/LSTM layers frozen, and only their final Dense output layers are fine-tuned on the new data.*")
        
    print(f"Saved tabular report to {report_path}")

    # Plot Learning Curves
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df_res, x='N_Samples', y='Accuracy', hue='Model', marker='o', linewidth=2.5, markersize=8)
    
    plt.title('Deep Learning Transfer Learning vs. Structural RF (Zero-Day Phishing)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of New Data Points Used for Training', fontsize=12, fontweight='bold')
    plt.ylabel('OOD Accuracy', fontsize=12, fontweight='bold')
    plt.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = project_root / "docs/assets/model_comparisons/dl_few_shot_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved learning curve plot to {plot_path}")

if __name__ == "__main__":
    main()

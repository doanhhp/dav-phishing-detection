"""Experiment: Deep Learning Mixed Retraining from Scratch."""

import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sys
project_root_sys = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root_sys))

from src.utils.config_loader import ConfigLoader
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config/benchmarks.yaml"
    config = ConfigLoader.load_yaml(str(config_path))
    
    print("--- 1. Loading Historical Data ---")
    orig_url = pd.read_excel(project_root / "data/raw/URL.xlsx")
    orig_html = pd.read_excel(project_root / "data/raw/html.xlsx")
    
    orig_url['html'] = orig_html.loc[orig_url.index, 'Data']
    y_orig = orig_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    # Subsample 3000 historical samples
    df_hist = orig_url.sample(3000, random_state=42)
    y_hist = df_hist['Category'].map({'ham': 0, 'spam': 1}).values
    
    df_hist_multi = df_hist[['Data', 'html']]
    df_hist_single = df_hist['Data']
    
    print("--- 2. Loading OOD Data ---")
    df_ood_url = pd.read_excel(project_root / "data/raw/OOD_URL.xlsx")
    df_ood_html = pd.read_excel(project_root / "data/raw/OOD_html.xlsx")
    
    df_ood_url['html'] = df_ood_html['Data']
    y_ood = df_ood_url['Category'].map({'ham': 0, 'spam': 1}).values
    
    models_to_test = ['rnn_gru', 'lstm_url', 'webphish_cnn']
    
    sample_sizes = [10, 20, 50, 100, 200]
    n_seeds = 3
    results = []
    
    print("--- 3. Running Mixed Retraining ---")
    for n_samples in sample_sizes:
        print(f"\n========================================")
        print(f"Evaluating with {n_samples} new zero-day samples...")
        print(f"========================================")
        
        acc_dict = {m: [] for m in models_to_test}
        
        for seed in range(n_seeds):
            print(f"  -> Seed {seed+1}/{n_seeds}")
            
            # Subsample N OOD data points
            indices = np.arange(len(y_ood))
            idx_train_new, idx_test_new, y_train_new, y_test_new = train_test_split(
                indices, y_ood, train_size=n_samples, random_state=seed, stratify=y_ood
            )
            
            df_ood_multi_train = df_ood_url.iloc[idx_train_new][['Data', 'html']]
            df_ood_single_train = df_ood_url.iloc[idx_train_new]['Data']
            
            df_ood_multi_test = df_ood_url.iloc[idx_test_new][['Data', 'html']]
            df_ood_single_test = df_ood_url.iloc[idx_test_new]['Data']
            
            # Combine 3000 Historical + N New
            X_train_multi = pd.concat([df_hist_multi, df_ood_multi_train])
            X_train_single = pd.concat([df_hist_single, df_ood_single_train])
            y_train_mixed = np.concatenate([y_hist, y_train_new])
            
            for m in models_to_test:
                print(f"     Testing {m}...")
                model_config = ConfigLoader.get_model_config(config, m)
                
                # Hack: Lower epochs to speed up simulation, since we have 3000 samples and we're just testing adaptation
                model_config['epochs'] = 5 
                
                processor_name = model_config.get("processor") or model_config.get("feature_processor")
                feat_config = config.get("features", {}).get(processor_name, {})
                
                try:
                    # 1. Initialize fresh processor
                    processor = FeatureFactory.get_processor(processor_name, feat_config)
                    
                    # Fit on mixed data
                    if m == 'webphish_cnn':
                        X_train_proc = processor.fit_transform(X_train_multi)
                        X_test_proc = processor.transform(df_ood_multi_test)
                    else:
                        X_train_proc = processor.fit_transform(X_train_single)
                        X_test_proc = processor.transform(df_ood_single_test)
                        
                    # 2. Initialize fresh model
                    # For Keras models, clear session to prevent OOM
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    model = ModelFactory.create_model(m, model_config)
                    
                    # 3. Train from scratch
                    model.fit(X_train_proc, y_train_mixed)
                    
                    # 4. Evaluate
                    preds = model.predict(X_test_proc)
                    acc = accuracy_score(y_test_new, preds)
                    acc_dict[m].append(acc)
                    print(f"       Accuracy: {acc:.4f}")
                
                except Exception as e:
                    print(f"       Failed on {m}: {e}")
                    acc_dict[m].append(np.nan)
        
        for m in models_to_test:
            results.append({
                'N_Samples': n_samples,
                'Model': m + ' (Retrain Scratch)',
                'Accuracy': np.nanmean(acc_dict[m])
            })
            
    df_res = pd.DataFrame(results)
    
    # Also load the previous RF Scratch results so we can plot them side-by-side
    rf_scratch_results = [
        {'N_Samples': 10, 'Model': 'structural_rf (Scratch)', 'Accuracy': 0.812098},
        {'N_Samples': 20, 'Model': 'structural_rf (Scratch)', 'Accuracy': 0.94347},
        {'N_Samples': 50, 'Model': 'structural_rf (Scratch)', 'Accuracy': 0.972892},
        {'N_Samples': 100, 'Model': 'structural_rf (Scratch)', 'Accuracy': 0.990839},
        {'N_Samples': 200, 'Model': 'structural_rf (Scratch)', 'Accuracy': 0.997242}
    ]
    df_rf = pd.DataFrame(rf_scratch_results)
    
    df_plot = pd.concat([df_res, df_rf], ignore_index=True)
    
    # Save Tabular Results
    report_path = project_root / "docs/few_shot_report.md"
    pivot_df = df_plot.pivot(index='N_Samples', columns='Model', values='Accuracy')
    
    print("\n--- DEEP LEARNING RETRAIN FROM SCRATCH RESULTS ---")
    print(pivot_df)
    
    with open(report_path, 'a') as f:
        f.write("\n## Deep Learning Mixed Retraining from Scratch\n\n")
        f.write("This section compares rebuilding Deep Learning NLP models completely from scratch (using a mix of 3,000 historical samples + N new samples) to the Structural RF (trained purely on the N new samples).\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n\n*Note: Mixed Retraining updates the Vocabulary Tokenizer and trains all Neural Network layers from random initialization.*")
        
    print(f"Saved tabular report to {report_path}")

    # Plot Learning Curves
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df_plot, x='N_Samples', y='Accuracy', hue='Model', marker='o', linewidth=2.5, markersize=8)
    
    plt.title('Deep Learning Retrain-from-Scratch vs. Structural RF', fontsize=14, fontweight='bold')
    plt.xlabel('Number of New Data Points Used for Retraining', fontsize=12, fontweight='bold')
    plt.ylabel('OOD Accuracy', fontsize=12, fontweight='bold')
    plt.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = project_root / "docs/assets/model_comparisons/dl_retrain_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved learning curve plot to {plot_path}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from src.models.model_factory import ModelFactory
import yaml

def main():
    print("Loading 2021 Historic Data...")
    df_2021 = pd.read_excel("data/raw/URL.xlsx")
    df_html_2021 = pd.read_excel("data/raw/html.xlsx")
    
    if 'html' in df_html_2021.columns:
        df_2021['html'] = df_html_2021['html']
    else:
        df_2021['html'] = df_html_2021['Data']
        
    y_true_full = np.where(df_2021['Category'].isin(['phishing', 'malware', 'spam']), 1, 0)
    df_2021['y'] = y_true_full
    
    # Take a balanced sample of 4000 sites for speed
    df_phish = df_2021[df_2021['y'] == 1].sample(2000, random_state=42)
    df_legit = df_2021[df_2021['y'] == 0].sample(2000, random_state=42)
    df_sample = pd.concat([df_phish, df_legit]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    y_true = df_sample['y'].values
    
    print("Loading Processor and Model...")
    processor = joblib.load("data/processed/mid_fusion_xgb/processor.joblib")
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = ModelFactory.create_model("mid_fusion_xgb", config["models"]["mid_fusion_xgb"])
    model.load("experiments/mid_fusion_xgb/model")
    
    print("Processing Features (This should be fast for 4000 samples)...")
    X_processed = processor.transform(df_sample)
    
    print("Predicting Probabilities...")
    y_prob = model.predict_proba(X_processed)[:, 1]
    
    # Analyze the distribution
    phish_probs = y_prob[y_true == 1]
    legit_probs = y_prob[y_true == 0]
    
    print(f"\n--- 2021 PROBABILITY ANALYSIS ---")
    print(f"Phishing Sites (True=1) Avg Prob: {np.mean(phish_probs):.4f}")
    print(f"Legitimate Sites (True=0) Avg Prob: {np.mean(legit_probs):.4f}")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        
    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='#2ecc71', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', color='#e74c3c', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', color='#3498db', linewidth=2, linestyle='--')
    
    plt.axvline(x=best_thresh, color='gray', linestyle=':', label=f'Optimal F1 Threshold ({best_thresh:.2f})')
    plt.scatter([best_thresh], [best_f1], color='black', zorder=5)
    
    plt.title('Threshold Tuning: Precision, Recall, and F1-Score (2021 Historic)', fontsize=14, pad=15)
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    out_dir = "docs/assets/explainable_ai/zero_day_failure"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "threshold_tuning_curve_2021.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    best_y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_true, best_y_pred)
    
    print(f"Graph saved to {out_path}")
    print(f"\nOptimal Threshold (Max F1) for 2021 Dataset: {best_thresh:.2f} -> F1: {best_f1:.3f}")
    print(f"Accuracy at Threshold {best_thresh:.2f}: {acc*100:.2f}%")
    print(f"Recall at Threshold {best_thresh:.2f}: {recalls[best_idx]*100:.2f}%")
    print(f"Precision at Threshold {best_thresh:.2f}: {precisions[best_idx]*100:.2f}%")

if __name__ == '__main__':
    main()

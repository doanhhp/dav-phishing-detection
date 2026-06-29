import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from src.models.model_factory import ModelFactory
import yaml

def main():
    print("Loading 2026 Zero-Day Data...")
    df_2026 = pd.read_excel("data/raw/OOD_URL.xlsx")
    df_html_2026 = pd.read_excel("data/raw/OOD_html.xlsx")
    df_2026['html'] = df_html_2026['Data']
    
    y_true = np.where(df_2026['Category'].isin(['phishing', 'malware', 'spam']), 1, 0)
    
    print("Loading Processor and Model...")
    processor = joblib.load("data/processed/mid_fusion_xgb/processor.joblib")
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = ModelFactory.create_model("mid_fusion_xgb", config["models"]["mid_fusion_xgb"])
    model.load("experiments/mid_fusion_xgb/model")
    
    print("Processing Features...")
    X_processed = processor.transform(df_2026)
    
    print("Predicting Probabilities...")
    y_prob = model.predict_proba(X_processed)[:, 1]
    
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        # Avoid division by zero warnings
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='#2ecc71', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', color='#e74c3c', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', color='#3498db', linewidth=2, linestyle='--')
    
    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.axvline(x=best_thresh, color='gray', linestyle=':', label=f'Optimal F1 Threshold ({best_thresh:.2f})')
    plt.scatter([best_thresh], [best_f1], color='black', zorder=5)
    
    plt.title('Threshold Tuning: Precision, Recall, and F1-Score (2026 Zero-Day)', fontsize=14, pad=15)
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    out_dir = "docs/assets/explainable_ai/zero_day_failure"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "threshold_tuning_curve.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    best_y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_true, best_y_pred)
    
    print(f"Graph saved to {out_path}")
    print(f"Optimal Threshold (Max F1): {best_thresh:.2f} -> F1: {best_f1:.3f}")
    print(f"Accuracy at Threshold {best_thresh:.2f}: {acc*100:.2f}%")
    print(f"Recall at Threshold {best_thresh:.2f}: {recalls[best_idx]*100:.2f}%")
    print(f"Precision at Threshold {best_thresh:.2f}: {precisions[best_idx]*100:.2f}%")

if __name__ == '__main__':
    main()

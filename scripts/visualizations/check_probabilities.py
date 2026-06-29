import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score

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
    
    # Analyze the distribution
    phish_probs = y_prob[y_true == 1]
    legit_probs = y_prob[y_true == 0]
    
    print(f"\n--- PROBABILITY ANALYSIS ---")
    print(f"Phishing Sites (True=1) Avg Prob: {np.mean(phish_probs):.4f}")
    print(f"Legitimate Sites (True=0) Avg Prob: {np.mean(legit_probs):.4f}")
    
    print(f"\nPhishing Sites Prob Distribution:")
    print(f"  < 0.1: {np.sum(phish_probs < 0.1)}")
    print(f"  0.1 - 0.3: {np.sum((phish_probs >= 0.1) & (phish_probs < 0.3))}")
    print(f"  0.3 - 0.5: {np.sum((phish_probs >= 0.3) & (phish_probs < 0.5))}")
    print(f"  0.5 - 0.7: {np.sum((phish_probs >= 0.5) & (phish_probs < 0.7))}")
    print(f"  > 0.7: {np.sum(phish_probs >= 0.7)}")
    
    # Test different thresholds
    print(f"\n--- RECALL AT DIFFERENT THRESHOLDS ---")
    for thresh in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        y_pred = (y_prob >= thresh).astype(int)
        recall = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(f"Threshold {thresh:.2f} -> Recall: {recall*100:.1f}%, Precision: {prec*100:.1f}%")
        print(f"  False Positives: {cm[0,1]}, True Positives: {cm[1,1]}")

if __name__ == '__main__':
    main()

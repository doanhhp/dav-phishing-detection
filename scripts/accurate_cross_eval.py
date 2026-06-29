import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.mid_fusion_xgb import MidFusionXGB
from src.features.structural import StructuralProcessor

def load_data(path, sample_n=None):
    df = pd.read_parquet(path)
    if 'Category' in df.columns:
        df = df.rename(columns={'Category': 'Class'})
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df

def main():
    print("Loading datasets (Sampling 5000 rows to speed up)...")
    main_df = load_data(project_root / 'data' / 'processed' / 'standardized' / 'main_dataset.parquet', sample_n=5000)
    ood_df = load_data(project_root / 'data' / 'processed' / 'standardized' / 'ood_dataset.parquet', sample_n=5000)
    phresh_df = load_data(project_root / 'data' / 'processed' / 'standardized' / 'phreshphish_dataset.parquet', sample_n=5000)
    
    print("Extracting features...")
    processor = StructuralProcessor({})
    
    # We only use structural features for Mid-Fusion XGBoost since it expects the preprocessed numeric tabular data
    X_main = processor.fit_transform(main_df)
    y_main = main_df['Class'].values
    
    X_ood = processor.transform(ood_df)
    y_ood = ood_df['Class'].values
    
    X_phresh = processor.transform(phresh_df)
    y_phresh = phresh_df['Class'].values
    
    print("\n--- EXPERIMENT 1: Train on Main (2021) -> Test on PhreshPhish (2026) ---")
    model1 = MidFusionXGB({})
    model1.fit(X_main, y_main)
    preds1 = model1.predict(X_phresh)
    
    acc1 = accuracy_score(y_phresh, preds1)
    rec1 = recall_score(y_phresh, preds1)
    pre1 = precision_score(y_phresh, preds1)
    f1_1 = f1_score(y_phresh, preds1)
    cm1 = confusion_matrix(y_phresh, preds1)
    print(f"Accuracy: {acc1:.4f}, Recall: {rec1:.4f}, Precision: {pre1:.4f}, F1: {f1_1:.4f}")
    print(f"Confusion Matrix:\n{cm1}")
    
    print("\n--- EXPERIMENT 2: Train on PhreshPhish (2026) -> Test on OOD (2024) ---")
    model2 = MidFusionXGB({})
    model2.fit(X_phresh, y_phresh)
    preds2 = model2.predict(X_ood)
    
    acc2 = accuracy_score(y_ood, preds2)
    rec2 = recall_score(y_ood, preds2)
    pre2 = precision_score(y_ood, preds2)
    f1_2 = f1_score(y_ood, preds2)
    cm2 = confusion_matrix(y_ood, preds2)
    print(f"Accuracy: {acc2:.4f}, Recall: {rec2:.4f}, Precision: {pre2:.4f}, F1: {f1_2:.4f}")
    print(f"Confusion Matrix:\n{cm2}")

if __name__ == "__main__":
    main()

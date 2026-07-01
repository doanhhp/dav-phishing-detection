import pandas as pd
import numpy as np
import joblib
import os
import sys
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def main():
    print("Loading 2021 Main Dataset for Training...")
    # Assuming the 40k one is phreshphish_40k.parquet based on previous scripts
    df_train = pd.read_parquet('data/raw/external/phreshphish_40k.parquet')
    
    print("Sampling 2000 rows to quickly train and test cross-domain generalization...")
    df_train = df_train.sample(2000, random_state=42)
    y_train = df_train['Category'].values
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_name = "webphish_cnn"
    model_config = config["models"][model_name]
    
    print("Initializing Multimodal Processor...")
    processor = FeatureFactory.get_processor("multimodal", model_config)
    
    print("Fitting Multimodal Processor on Training Data...")
    X_train = processor.fit_transform(df_train[['Data', 'html']])
    
    print("Creating and Training WebPhish CNN...")
    # Temporarily override epochs to 3 for speed
    model_config["epochs"] = 3
    model = ModelFactory.create_model(model_name, model_config)
    model.fit(X_train, y_train)
    
    print("Loading 2026 OOD Testing Data...")
    df_test = pd.read_parquet('data/processed/standardized/ood_dataset.parquet')
    y_test = df_test['Category'].values
    
    print("Transforming Test Data...")
    X_test = processor.transform(df_test[['Data', 'html']])
    
    print("Evaluating WebPhish CNN on OOD...")
    y_prob = model.predict_proba(X_test)
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- WebPhish CNN Results on 2026 OOD ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()

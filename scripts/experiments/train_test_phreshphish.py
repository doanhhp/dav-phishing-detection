import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import logging

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading PhreshPhish dataset...")
    df = pd.read_parquet('data/raw/external/phreshphish_40k.parquet')
    
    # Category is already 1/0, and structural.py expects 'Data' and 'html' columns
    
    # Category is already 1/0
    y = df['Category'].values
    
    # Split data 80/20
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set size: {len(df_train)}")
    logger.info(f"Test set size: {len(df_test)}")
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    logger.info("Initializing structural feature processor...")
    processor = FeatureFactory.get_processor('structural', {})
    
    logger.info("Processing training features...")
    X_train = processor.fit_transform(df_train)
    
    logger.info("Processing test features...")
    X_test = processor.transform(df_test)
    
    logger.info("Creating and training mid_fusion_xgb model...")
    model = ModelFactory.create_model("mid_fusion_xgb", config["models"]["mid_fusion_xgb"])
    model.fit(X_train, y_train)
    
    logger.info("Evaluating on PhreshPhish Test Set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"--- Results on PhreshPhish (In-Distribution) ---")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

if __name__ == '__main__':
    main()

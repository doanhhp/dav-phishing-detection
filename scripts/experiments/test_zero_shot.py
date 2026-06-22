import argparse
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import time
import os
import yaml
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pipeline import load_data
from src.models.model_factory import ModelFactory
from src.features.structural import StructuralProcessor
from src.features.multimodal import MultimodalProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_zero_shot(model_name, config_path, url_path, html_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_config = config["models"][model_name]
    
    logger.info(f"Loading data from {url_path} and {html_path}")
    df, y = load_data(url_path, html_path)
    
    X = df[['Data', 'html']] if model_name in ["webphish_cnn", "egso_cnn", "structural_dnn", "structural_rf", "structural_gb", "structural_xgb", "hybrid_nn", "structural_stacking"] else df['Data']
    
    processor_name = model_config.get("processor") or model_config.get("feature_processor")
    
    # Load processor
    logger.info(f"Loading saved feature processor for {model_name}")
    processor_save_path = os.path.join("data", "processed", f"{model_name}", "processor.joblib")
    
    if os.path.exists(processor_save_path):
        import joblib
        processor = joblib.load(processor_save_path)
        logger.info(f"Processing test features...")
        X_processed = processor.transform(X)
    else:
        logger.warning(f"Processor not found at {processor_save_path}. Cannot perform zero-shot.")
        return
        
    # Load model
    logger.info(f"Loading saved model {model_name}")
    model = ModelFactory.create_model(model_name, model_config)
    
    if model_name in ["structural_xgb", "structural_stacking"]:
        model_load_path = os.path.join("experiments", model_name, "model")
    else:
        model_load_path = os.path.join("experiments", model_name, "model.h5")
        
    model.load(model_load_path)
    
    logger.info("Evaluating...")
    start_time = time.time()
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_processed)
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_processed)
        y_prob = y_pred
    inference_time = time.time() - start_time
    latency_ms = (inference_time / len(X_processed)) * 1000
        
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    roc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred).tolist()
    
    logger.info(f"--- Results for {model_name} (Zero-Shot) ---")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc:.4f}")
    logger.info(f"Inference Latency: {latency_ms:.4f} ms/sample")
    logger.info(f"Confusion Matrix: {cm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("config")
    parser.add_argument("--url_path", required=True)
    parser.add_argument("--html_path", required=True)
    args = parser.parse_args()
    
    run_zero_shot(args.model, args.config, args.url_path, args.html_path)

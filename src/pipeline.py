"""Main training and evaluation pipeline."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import Metrics

logger = setup_logger(__name__)

def load_data(url_path: str, html_path: str = None, samples: int = None):
    """Load and merge URL and HTML data."""
    logger.info(f"Loading URL data from {url_path}")
    df_url = pd.read_excel(url_path)
    
    if samples:
        df_url = df_url.sample(min(samples, len(df_url)), random_state=42)
        
    if html_path:
        logger.info(f"Loading HTML data from {html_path}")
        df_html = pd.read_excel(html_path)
        # Assuming they are aligned by index
        df_url['html'] = df_html.loc[df_url.index, 'Data']
    
    # Map categories to labels
    label_map = {'ham': 0, 'spam': 1}
    y = df_url['Category'].map(label_map).values
    
    return df_url, y

def run_experiment(model_name: str, config_path: str, samples: int = None):
    """
    Run a single model experiment.
    """
    logger.info(f"--- Starting experiment for {model_name} ---")

    # 1. Load configuration
    config = ConfigLoader.load_yaml(config_path)
    model_config = ConfigLoader.get_model_config(config, model_name)
    global_config = ConfigLoader.get_global_config(config)
    
    # Merge global configs into model config
    model_config.update({
        "batch_size": global_config.get("batch_size", 32),
        "epochs": global_config.get("epochs", 10),
        "random_seed": global_config.get("random_seed", 42)
    })

    # 2. Load dataset
    url_path = "data/raw/URL.xlsx"
    html_path = "data/raw/html.xlsx" if model_name == "webphish_cnn" else None
    
    df, y = load_data(url_path, html_path, samples=samples)
    
    # 3. Split data
    X = df['Data'] if model_name != "webphish_cnn" else df[['Data', 'html']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=global_config.get("test_split", 0.2),
        random_state=global_config.get("random_seed", 42),
        stratify=y
    )

    # 4. Get feature processor
    processor_name = model_config.get("feature_processor")
    logger.info(f"Initializing feature processor: {processor_name}")
    
    # Merge feature configs
    feat_config = config.get("features", {}).get(processor_name, {})
    processor = FeatureFactory.get_processor(processor_name, feat_config)
    
    # 5. Process features
    logger.info("Processing features...")
    X_train_processed = processor.fit_transform(X_train)
    X_test_processed = processor.transform(X_test)
    
    # 6. Create and train model
    logger.info(f"Creating model: {model_name}")
    model = ModelFactory.create_model(model_name, model_config)
    
    logger.info("Training model...")
    model.fit(X_train_processed, y_train)
    
    # 7. Make predictions and calculate metrics
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)
    
    # Handle probability format (take probability of class 1)
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba
        
    metrics = Metrics.calculate_all(y_test, y_pred, y_proba_pos)
    logger.info(f"Metrics: {metrics}")
    
    # 8. Save results
    exp_dir = Path(f"experiments/{model_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    if "sklearn" in model_config.get("type", ""):
        save_path = exp_dir / "model.joblib"
    elif "pytorch" in model_config.get("type", ""):
        save_path = exp_dir / "model.pth"
    else:
        save_path = exp_dir / "model.h5"
        
    model.save(str(save_path))
    
    logger.info(f"--- Experiment for {model_name} completed ---")
    return metrics

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run phishing detection experiment")
    parser.add_argument("model", help="Model name")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to use")
    
    args = parser.parse_args()
    run_experiment(args.model, args.config, samples=args.samples)

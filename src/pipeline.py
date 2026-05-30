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

    # 3. Split data & CV Setup
    X = df[['Data', 'html']] if model_name in ["webphish_cnn", "egso_cnn"] else df['Data']
    processor_name = model_config.get("feature_processor")
    
    if cv:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=global_config.get("random_seed", 42))
        all_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"--- Fold {fold + 1}/{cv} ---")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            feat_config = config.get("features", {}).get(processor_name, {})
            processor = FeatureFactory.get_processor(processor_name, feat_config)
            X_train_processed = processor.fit_transform(X_train)
            X_test_processed = processor.transform(X_test)
            
            model = ModelFactory.create_model(model_name, model_config)
            model.fit(X_train_processed, y_train)
            
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)
            y_proba_pos = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
            
            metrics = Metrics.calculate_all(y_test, y_pred, y_proba_pos)
            all_metrics.append(metrics)
            
        # Average metrics
        metrics = {}
        for k in all_metrics[0].keys():
            if isinstance(all_metrics[0][k], (int, float)):
                metrics[k] = float(np.mean([m[k] for m in all_metrics]))
            else:
                metrics[k] = all_metrics[0][k]
        logger.info(f"CV Metrics: {metrics}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=global_config.get("test_split", 0.2),
            random_state=global_config.get("random_seed", 42),
            stratify=y
        )

        logger.info(f"Initializing feature processor: {processor_name}")
        feat_config = config.get("features", {}).get(processor_name, {})
        processor = FeatureFactory.get_processor(processor_name, feat_config)
        
        logger.info("Processing features...")
        X_train_processed = processor.fit_transform(X_train)
        X_test_processed = processor.transform(X_test)
        
        logger.info(f"Creating model: {model_name}")
        model = ModelFactory.create_model(model_name, model_config)
        
        logger.info("Training model...")
        model.fit(X_train_processed, y_train)
        
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)
        
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
        
    try:
        model.save(str(save_path))
    except Exception as e:
        logger.error(f"Could not save model: {e}")
    
    logger.info(f"--- Experiment for {model_name} completed ---")
    return metrics

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run phishing detection experiment")
    parser.add_argument("model", help="Model name")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--cv", type=int, default=None, help="Number of CV folds")
    
    args = parser.parse_args()
    run_experiment(args.model, args.config, samples=args.samples, cv=args.cv)

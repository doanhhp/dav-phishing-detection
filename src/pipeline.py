"""Main training and evaluation pipeline."""

import json
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import Metrics

logger = setup_logger(__name__)

def run_experiment(model_name: str, config_path: str, dataset_path: str = None):
    """
    Run a single model experiment.

    Args:
        model_name: Name of the model to train
        config_path: Path to configuration YAML file
        dataset_path: Path to dataset (optional, will be added later)
    """
    logger.info(f"Running experiment for {model_name}")

    # Load configuration
    config = ConfigLoader.load_yaml(config_path)
    model_config = ConfigLoader.get_model_config(config, model_name)

    logger.info(f"Model config loaded: {model_config}")

    # TODO: Implement full pipeline:
    # 1. Load dataset from dataset_path
    # 2. Get feature processor
    # 3. Process features
    # 4. Create model
    # 5. Train model
    # 6. Make predictions
    # 7. Calculate metrics
    # 8. Save results

    logger.info(f"Experiment for {model_name} completed")

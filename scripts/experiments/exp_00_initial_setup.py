#!/usr/bin/env python3
"""
Modular Benchmarking Architecture Setup Script

Creates the complete directory structure, initializes __init__.py files,
and generates template files for the phishing detection benchmarking system.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Define the complete directory structure
DIRECTORIES = [
    "config",
    "data/raw",
    "data/processed/hybrid_svm_knn",
    "data/processed/lstm_url",
    "data/processed/webphish_cnn",
    "data/processed/egso_cnn",
    "data/processed/rnn_gru",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/utils",
    "experiments/hybrid_svm_knn",
    "experiments/lstm_url",
    "experiments/webphish_cnn",
    "experiments/egso_cnn",
    "experiments/rnn_gru",
    "reports/comparison",
    "reports/individual",
    "notebooks",
    "tests",
]

# Define files with their content (key: file path, value: content)
INIT_FILES = {
    "src/__init__.py": '',
    "src/features/__init__.py": 'from .factory import FeatureFactory\n\n__all__ = ["FeatureFactory"]\n',
    "src/models/__init__.py": 'from .model_factory import ModelFactory\n\n__all__ = ["ModelFactory"]\n',
    "src/evaluation/__init__.py": 'from .evaluate import GlobalEvaluator\nfrom .metrics import Metrics\nfrom .visualizer import Visualizer\n\n__all__ = ["GlobalEvaluator", "Metrics", "Visualizer"]\n',
    "src/utils/__init__.py": 'from .config_loader import ConfigLoader\nfrom .logger import setup_logger\n\n__all__ = ["ConfigLoader", "setup_logger"]\n',
    "tests/__init__.py": '',
}

STUB_FILES = {
    "src/features/factory.py": """\"\"\"Feature processor factory for modular feature engineering.\"\"\"

class FeatureFactory:
    \"\"\"Factory for instantiating feature processors based on model type.\"\"\"

    @staticmethod
    def get_processor(processor_name: str, config: dict):
        \"\"\"
        Get a feature processor instance.

        Args:
            processor_name: Type of processor ('manual', 'sequential', 'multimodal', 'tfidf_svd', 'rnn_gru')
            config: Configuration dictionary

        Returns:
            Feature processor instance
        \"\"\"
        from .manual import ManualFeatureProcessor
        from .sequential import SequentialTokenProcessor
        from .multimodal import MultimodalProcessor
        from .tfidf_svd import TfidfSvdProcessor
        from .rnn_gru import RnnGruProcessor

        processors = {
            "manual": ManualFeatureProcessor,
            "sequential": SequentialTokenProcessor,
            "multimodal": MultimodalProcessor,
            "tfidf_svd": TfidfSvdProcessor,
            "rnn_gru": RnnGruProcessor,
        }

        if processor_name not in processors:
            raise ValueError(f"Unknown processor: {processor_name}. Available: {list(processors.keys())}")

        return processors[processor_name](config)
""",

    "src/features/manual.py": """\"\"\"Manual feature processor for SVM+KNN models.\"\"\"

class ManualFeatureProcessor:
    \"\"\"Processes manual features for hybrid SVM+KNN model.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False

    def fit_transform(self, X, y=None):
        \"\"\"Fit and transform features.\"\"\"
        self.fitted = True
        # TODO: Implement manual feature extraction
        return X

    def transform(self, X):
        \"\"\"Transform features.\"\"\"
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        # TODO: Implement manual feature transformation
        return X
""",

    "src/features/sequential.py": """\"\"\"Sequential token processor for LSTM models.\"\"\"

class SequentialTokenProcessor:
    \"\"\"Processes character sequences for LSTM URL-Only model.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.tokenizer = None

    def fit_transform(self, X, y=None):
        \"\"\"Fit tokenizer and transform sequences.\"\"\"
        self.fitted = True
        # TODO: Implement sequence tokenization and padding
        return X

    def transform(self, X):
        \"\"\"Transform sequences.\"\"\"
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        # TODO: Implement sequence transformation
        return X
""",

    "src/features/multimodal.py": """\"\"\"Multimodal feature processor for WebPhish CNN models.\"\"\"

class MultimodalProcessor:
    \"\"\"Processes URL and HTML features for WebPhish CNN model.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False

    def fit_transform(self, X, y=None):
        \"\"\"Fit and transform multimodal features.\"\"\"
        self.fitted = True
        # TODO: Implement multimodal feature extraction (URL + HTML)
        return X

    def transform(self, X):
        \"\"\"Transform multimodal features.\"\"\"
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        # TODO: Implement multimodal feature transformation
        return X
""",

    "src/features/tfidf_svd.py": """\"\"\"TF-IDF + SVD feature processor for EGSO-CNN models.\"\"\"

class TfidfSvdProcessor:
    \"\"\"Processes features using TF-IDF and SVD dimensionality reduction.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.tfidf = None
        self.svd = None

    def fit_transform(self, X, y=None):
        \"\"\"Fit TF-IDF and SVD, then transform features.\"\"\"
        self.fitted = True
        # TODO: Implement TF-IDF and SVD fitting
        return X

    def transform(self, X):
        \"\"\"Transform features using fitted TF-IDF and SVD.\"\"\"
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        # TODO: Implement TF-IDF and SVD transformation
        return X
""",

    "src/features/rnn_gru.py": """\"\"\"Sequential token processor for RNN/GRU models.\"\"\"

class RnnGruProcessor:
    \"\"\"Processes character sequences for RNN GRU model.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.tokenizer = None

    def fit_transform(self, X, y=None):
        \"\"\"Fit tokenizer and transform sequences.\"\"\"
        self.fitted = True
        # TODO: Implement sequence tokenization and padding
        return X

    def transform(self, X):
        \"\"\"Transform sequences.\"\"\"
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        # TODO: Implement sequence transformation
        return X
""",

    "src/models/base.py": """\"\"\"Base model class for all benchmarking models.\"\"\"

from abc import ABC, abstractmethod

class BaseModel(ABC):
    \"\"\"Abstract base class for all phishing detection models.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.trained = False

    @abstractmethod
    def fit(self, X, y):
        \"\"\"Train the model.\"\"\"
        pass

    @abstractmethod
    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        pass

    @abstractmethod
    def predict_proba(self, X):
        \"\"\"Predict class probabilities.\"\"\"
        pass

    def save(self, path: str):
        \"\"\"Save the model.\"\"\"
        raise NotImplementedError("Model saving must be implemented in subclass")

    def load(self, path: str):
        \"\"\"Load a saved model.\"\"\"
        raise NotImplementedError("Model loading must be implemented in subclass")
""",

    "src/models/model_factory.py": """\"\"\"Model factory for instantiating different phishing detection models.\"\"\"

class ModelFactory:
    \"\"\"Factory for creating model instances.\"\"\"

    @staticmethod
    def create_model(model_name: str, config: dict):
        \"\"\"
        Create a model instance.

        Args:
            model_name: Name of model ('hybrid_svm_knn', 'lstm_url', 'webphish_cnn', 'egso_cnn', 'rnn_gru')
            config: Configuration dictionary

        Returns:
            Model instance
        \"\"\"
        from .hybrid_svm_knn import SVM_KNN
        from .lstm_url import LSTM_URL
        from .webphish_cnn import WebPhish_CNN
        from .egso_cnn import EGSO_CNN
        from .rnn_gru import RNN_GRU

        models = {
            "hybrid_svm_knn": SVM_KNN,
            "lstm_url": LSTM_URL,
            "webphish_cnn": WebPhish_CNN,
            "egso_cnn": EGSO_CNN,
            "rnn_gru": RNN_GRU,
        }

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

        return models[model_name](config)
""",

    "src/models/hybrid_svm_knn.py": """\"\"\"Hybrid SVM+KNN phishing detection model.\"\"\"

from .base import BaseModel

class SVM_KNN(BaseModel):
    \"\"\"Hybrid SVM+KNN model for phishing detection using manual features.\"\"\"

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: Initialize SVM and KNN models

    def fit(self, X, y):
        \"\"\"Train the SVM+KNN model.\"\"\"
        self.trained = True
        # TODO: Implement training logic

    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement prediction logic
        return None

    def predict_proba(self, X):
        \"\"\"Predict class probabilities.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement probability prediction
        return None
""",

    "src/models/lstm_url.py": """\"\"\"LSTM URL-only phishing detection model.\"\"\"

from .base import BaseModel

class LSTM_URL(BaseModel):
    \"\"\"LSTM model for phishing detection using URL sequences only.\"\"\"

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: Initialize LSTM model

    def fit(self, X, y):
        \"\"\"Train the LSTM model.\"\"\"
        self.trained = True
        # TODO: Implement training logic

    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement prediction logic
        return None

    def predict_proba(self, X):
        \"\"\"Predict class probabilities.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement probability prediction
        return None
""",

    "src/models/webphish_cnn.py": """\"\"\"WebPhish CNN phishing detection model.\"\"\"

from .base import BaseModel

class WebPhish_CNN(BaseModel):
    \"\"\"Multi-modal CNN model for phishing detection using URL and HTML.\"\"\"

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: Initialize CNN model

    def fit(self, X, y):
        \"\"\"Train the CNN model.\"\"\"
        self.trained = True
        # TODO: Implement training logic

    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement prediction logic
        return None

    def predict_proba(self, X):
        \"\"\"Predict class probabilities.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement probability prediction
        return None
""",

    "src/models/egso_cnn.py": """\"\"\"EGSO-CNN phishing detection model (2025).\"\"\"

from .base import BaseModel

class EGSO_CNN(BaseModel):
    \"\"\"Optimized CNN model with TF-IDF and feature reduction.\"\"\"

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: Initialize EGSO-CNN model

    def fit(self, X, y):
        \"\"\"Train the EGSO-CNN model.\"\"\"
        self.trained = True
        # TODO: Implement training logic

    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement prediction logic
        return None

    def predict_proba(self, X):
        \"\"\"Predict class probabilities.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement probability prediction
        return None
""",

    "src/models/rnn_gru.py": """\"\"\"RNN GRU phishing detection model.\"\"\"

from .base import BaseModel

class RNN_GRU(BaseModel):
    \"\"\"RNN GRU model for phishing detection.\"\"\"

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: Initialize RNN GRU model

    def fit(self, X, y):
        \"\"\"Train the RNN GRU model.\"\"\"
        self.trained = True
        # TODO: Implement training logic

    def predict(self, X):
        \"\"\"Make predictions.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement prediction logic
        return None

    def predict_proba(self, X):
        \"\"\"Predict class probabilities.\"\"\"
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        # TODO: Implement probability prediction
        return None
""",

    "src/evaluation/metrics.py": """\"\"\"Metrics calculation for model evaluation.\"\"\"

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

class Metrics:
    \"\"\"Calculate evaluation metrics.\"\"\"

    @staticmethod
    def calculate_all(y_true, y_pred, y_proba=None):
        \"\"\"
        Calculate all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary of metrics
        \"\"\"
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = None

        return metrics
""",

    "src/evaluation/visualizer.py": """\"\"\"Visualization utilities for evaluation results.\"\"\"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

class Visualizer:
    \"\"\"Generate visualizations for model evaluation.\"\"\"

    @staticmethod
    def plot_roc_curves(results_dict, output_path: str):
        \"\"\"Plot ROC curves for all models.\"\"\"
        plt.figure(figsize=(10, 8))
        for model_name, data in results_dict.items():
            if 'fpr' in data and 'tpr' in data:
                plt.plot(data['fpr'], data['tpr'], label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrices(results_dict, output_path: str):
        \"\"\"Plot confusion matrices for all models.\"\"\"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (model_name, data) in enumerate(results_dict.items()):
            if idx < 4 and 'confusion_matrix' in data:
                sns.heatmap(data['confusion_matrix'], ax=axes[idx], cmap='Blues')
                axes[idx].set_title(model_name)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_metrics_heatmap(results_df, output_path: str):
        \"\"\"Plot heatmap of all metrics.\"\"\"
        plt.figure(figsize=(10, 6))
        sns.heatmap(results_df, annot=True, cmap='YlOrRd')
        plt.title('Model Performance Metrics Comparison')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
""",

    "src/evaluation/evaluate.py": """\"\"\"Global evaluator for comparing all models.\"\"\"

import json
import pandas as pd
from pathlib import Path
from .visualizer import Visualizer

class GlobalEvaluator:
    \"\"\"Evaluate and compare all model experiments.\"\"\"

    def __init__(self, experiments_dir: str, output_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_leaderboard(self) -> pd.DataFrame:
        \"\"\"Generate performance leaderboard from all experiments.\"\"\"
        results = []

        for exp_dir in self.experiments_dir.glob("*/"):
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    metrics = json.load(f)
                results.append({
                    'Model': exp_dir.name,
                    **metrics
                })

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "leaderboard.csv", index=False)
        return df

    def generate_all_visualizations(self, results_dict):
        \"\"\"Generate all comparison visualizations.\"\"\"
        Visualizer.plot_roc_curves(results_dict, str(self.output_dir / "roc_curves.png"))
        Visualizer.plot_confusion_matrices(results_dict, str(self.output_dir / "confusion_matrices.png"))
""",

    "src/utils/config_loader.py": """\"\"\"Configuration loading utilities.\"\"\"

import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    \"\"\"Load and manage YAML configuration files.\"\"\"

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        \"\"\"Load YAML configuration file.\"\"\"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        \"\"\"Get model-specific configuration.\"\"\"
        return config.get('models', {}).get(model_name, {})

    @staticmethod
    def get_global_config(config: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Get global configuration settings.\"\"\"
        return config.get('global', {})
""",

    "src/utils/logger.py": """\"\"\"Logging utility setup.\"\"\"

import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    \"\"\"Set up a logger with optional file output.\"\"\"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
""",

    "src/pipeline.py": """\"\"\"Main training and evaluation pipeline.\"\"\"

import json
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import Metrics

logger = setup_logger(__name__)

def run_experiment(model_name: str, config_path: str, dataset_path: str = None):
    \"\"\"
    Run a single model experiment.

    Args:
        model_name: Name of the model to train
        config_path: Path to configuration YAML file
        dataset_path: Path to dataset (optional, will be added later)
    \"\"\"
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
""",

    "tests/test_model_factory.py": """\"\"\"Tests for model factory.\"\"\"

import pytest
from src.models.model_factory import ModelFactory

def test_create_hybrid_svm_knn():
    \"\"\"Test creation of SVM+KNN model.\"\"\"
    config = {}
    model = ModelFactory.create_model("hybrid_svm_knn", config)
    assert model is not None

def test_create_lstm_url():
    \"\"\"Test creation of LSTM model.\"\"\"
    config = {}
    model = ModelFactory.create_model("lstm_url", config)
    assert model is not None

def test_invalid_model():
    \"\"\"Test error on invalid model name.\"\"\"
    with pytest.raises(ValueError):
        ModelFactory.create_model("invalid_model", {})
""",

    "tests/test_feature_processors.py": """\"\"\"Tests for feature processors.\"\"\"

import pytest
from src.features.factory import FeatureFactory

def test_create_manual_processor():
    \"\"\"Test creation of manual feature processor.\"\"\"
    config = {}
    processor = FeatureFactory.get_processor("manual", config)
    assert processor is not None

def test_create_sequential_processor():
    \"\"\"Test creation of sequential processor.\"\"\"
    config = {}
    processor = FeatureFactory.get_processor("sequential", config)
    assert processor is not None

def test_invalid_processor():
    \"\"\"Test error on invalid processor name.\"\"\"
    with pytest.raises(ValueError):
        FeatureFactory.get_processor("invalid_processor", {})
""",

    "tests/test_evaluation.py": """\"\"\"Tests for evaluation metrics.\"\"\"

import numpy as np
from src.evaluation.metrics import Metrics

def test_metrics_calculation():
    \"\"\"Test metrics calculation.\"\"\"
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    metrics = Metrics.calculate_all(y_true, y_pred)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
""",
}

CONFIG_YAML = """# Modular Benchmarking Architecture Configuration

# Global settings
global:
  batch_size: 32
  epochs: 50
  validation_split: 0.2
  test_split: 0.1
  random_seed: 42
  output_dir: reports/comparison

# Model-specific hyperparameters
models:
  hybrid_svm_knn:
    type: "sklearn"
    svm_kernel: "rbf"
    svm_C: 1.0
    knn_neighbors: 5
    feature_processor: "manual"

  lstm_url:
    type: "tensorflow"
    embedding_dim: 128
    lstm_units: 64
    dropout: 0.3
    feature_processor: "sequential"

  webphish_cnn:
    type: "tensorflow"
    filters: [32, 64, 128]
    kernel_size: 3
    dropout: 0.4
    feature_processor: "multimodal"

  egso_cnn:
    type: "tensorflow"
    filters: [64, 128]
    kernel_size: 3
    tfidf_ngram: [1, 2]
    svd_components: 100
    feature_processor: "tfidf_svd"

# Feature processing configuration
features:
  manual:
    n_features: 48
  sequential:
    max_seq_length: 256
    vocab_size: 128
  multimodal:
    url_max_length: 200
    html_max_length: 5000
  tfidf_svd:
    max_features: 5000
    svd_components: 100
"""

REQUIREMENTS = """# Modular Benchmarking Architecture - Requirements

# Core ML frameworks
scikit-learn==1.3.0
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3

# Configuration management
PyYAML==6.0

# Visualization and reporting
matplotlib==3.7.2
seaborn==0.12.2

# Development and testing
pytest==7.4.0
pytest-cov==4.1.0

# Utilities
tqdm==4.65.0
"""

def create_directories():
    """Create all required directories."""
    for directory in DIRECTORIES:
        dir_path = BASE_DIR / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[+] Created: {directory}")

def create_init_files():
    """Create __init__.py files."""
    for file_path, content in INIT_FILES.items():
        full_path = BASE_DIR / file_path
        full_path.write_text(content)
        print(f"[+] Created: {file_path}")

def create_stub_files():
    """Create model and feature stub files."""
    for file_path, content in STUB_FILES.items():
        full_path = BASE_DIR / file_path
        full_path.write_text(content)
        print(f"[+] Created: {file_path}")

def create_config_files():
    """Create configuration files."""
    # Create benchmarks.yaml
    config_path = BASE_DIR / "config" / "benchmarks.yaml"
    config_path.write_text(CONFIG_YAML)
    print(f"[+] Created: config/benchmarks.yaml")

    # Create requirements.txt
    req_path = BASE_DIR / "requirements.txt"
    req_path.write_text(REQUIREMENTS)
    print(f"[+] Created: requirements.txt")

    # README will be created separately due to encoding issues

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data and outputs
data/raw/
data/processed/
experiments/*/weights.*
experiments/*/predictions.json
reports/comparison/*.csv
reports/comparison/*.png
*.log

# OS
.DS_Store
Thumbs.db
"""
    gitignore_path = BASE_DIR / ".gitignore"
    gitignore_path.write_text(gitignore_content)
    print(f"[+] Created: .gitignore")

def main():
    """Run the complete setup."""
    print("=" * 60)
    print("Modular Benchmarking Architecture - Setup")
    print("=" * 60)

    try:
        print("\n1. Creating directories...")
        create_directories()

        print("\n2. Creating __init__.py files...")
        create_init_files()

        print("\n3. Creating stub files...")
        create_stub_files()

        print("\n4. Creating configuration files...")
        create_config_files()

        print("\n5. Creating .gitignore...")
        create_gitignore()

        print("\n" + "=" * 60)
        print("[SUCCESS] Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Review config/benchmarks.yaml")
        print("3. Integrate your model implementations")
        print("4. Run experiments with src/pipeline.py")
        print("5. Generate comparisons with src/evaluation/evaluate.py")

    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

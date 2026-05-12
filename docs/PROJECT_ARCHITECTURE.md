# Project Architecture & Setup

## Overview

This benchmarking framework compares five phishing detection methods using a modular architecture with factory patterns, centralized evaluation, and configuration-driven model selection.

## Directory Structure

```
phishing-detection/
├── config/
│   └── benchmarks.yaml          # Central configuration for all models
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Feature-processed data by model
├── src/
│   ├── features/                # Feature processors (5 types)
│   │   ├── factory.py
│   │   ├── manual.py
│   │   ├── sequential.py
│   │   ├── rnn_gru.py
│   │   ├── multimodal.py
│   │   └── tfidf_svd.py
│   ├── models/                  # Model implementations (5 models)
│   │   ├── base.py
│   │   ├── model_factory.py
│   │   ├── hybrid_svm_knn.py
│   │   ├── lstm_url.py
│   │   ├── rnn_gru.py
│   │   ├── webphish_cnn.py
│   │   └── egso_cnn.py
│   ├── evaluation/              # Metrics & visualization
│   │   ├── metrics.py
│   │   ├── visualizer.py
│   │   └── evaluate.py
│   ├── utils/                   # Utilities
│   │   ├── config_loader.py
│   │   └── logger.py
│   └── pipeline.py              # Main training pipeline
├── experiments/                 # Model results by experiment
├── reports/                     # Visualizations & comparisons
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── tests/                       # Unit tests
├── scripts/
│   └── setup_benchmark.py       # Project initialization
├── README.md                    # Getting started guide
├── requirements.txt             # Python dependencies
└── docs/
    ├── MODELS.md               # Complete model reference
    └── PROJECT_ARCHITECTURE.md # This file
```

## Design Principles

### 1. **Modularity**
- Each model is independent and self-contained
- Feature processors are swappable and reusable
- Factory pattern decouples dependencies

### 2. **Configurability**
- Central YAML controls all settings
- No hardcoded values in code
- Easy hyperparameter tuning

### 3. **Scalability**
- New models can be added without modifying existing code
- New features can be processed in parallel
- Results aggregated in global evaluation

### 4. **Testability**
- Unit tests for factories, metrics, and features
- Isolated components are easy to test
- Mock data for testing workflows

## Key Components

### Configuration Management (`config/benchmarks.yaml`)

- **Global Settings**: batch_size, epochs, validation split, random seed
- **Model Parameters**: Framework-specific hyperparameters for each model
- **Feature Configuration**: Feature extraction parameters per processor

### Factory Pattern

**Model Factory** (`src/models/model_factory.py`)
```python
ModelFactory.create_model("hybrid_svm_knn", config)
ModelFactory.create_model("lstm_url", config)
ModelFactory.create_model("rnn_gru", config)
ModelFactory.create_model("webphish_cnn", config)
ModelFactory.create_model("egso_cnn", config)
```

**Feature Factory** (`src/features/factory.py`)
```python
FeatureFactory.get_processor("manual", config)
FeatureFactory.get_processor("sequential", config)
FeatureFactory.get_processor("rnn_gru", config)
FeatureFactory.get_processor("multimodal", config)
FeatureFactory.get_processor("tfidf_svd", config)
```

### Base Model Pattern

All models inherit from `BaseModel` and implement:
```python
class BaseModel:
    def fit(self, X, y): pass
    def predict(self, X): pass
    def predict_proba(self, X): pass
    def evaluate(self, X, y): pass
```

This ensures consistency and enables interchangeable usage.

### Pipeline & Evaluation

**Main Pipeline** (`src/pipeline.py`)
- Orchestrates training and evaluation for a single model
- Handles data loading, preprocessing, and result storage
- Usage: `python -m src.pipeline <model_name> config/benchmarks.yaml`

**Global Evaluator** (`src/evaluation/evaluate.py`)
- Aggregates results from all models
- Generates comparison visualizations
- Creates leaderboards and reports

## Workflow

### Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure models** (edit `config/benchmarks.yaml`)

3. **Train single model**
   ```bash
   python -m src.pipeline hybrid_svm_knn config/benchmarks.yaml
   ```

4. **Train all models**
   ```bash
   for model in hybrid_svm_knn lstm_url rnn_gru webphish_cnn egso_cnn; do
       python -m src.pipeline $model config/benchmarks.yaml
   done
   ```

5. **Generate comparison report**
   ```bash
   python -m src.evaluation.evaluate experiments/ reports/comparison/
   ```

## Testing

Run the test suite:
```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

Tests include:
- Model factory instantiation
- Feature processor pipeline
- Evaluation metrics calculation

## Development

### Adding a New Model

1. Create model file in `src/models/new_model.py`
2. Inherit from `BaseModel`
3. Implement required methods
4. Add to `ModelFactory.create_model()`
5. Add configuration to `config/benchmarks.yaml`

### Adding a New Feature Processor

1. Create processor in `src/features/new_processor.py`
2. Implement `process()` method
3. Add to `FeatureFactory.get_processor()`
4. Add configuration to `config/benchmarks.yaml`

## Files Generated by This Session

- `src/models/base.py` - Abstract base class
- `src/models/model_factory.py` - Model instantiation factory
- `src/features/factory.py` - Feature processor factory
- `config/benchmarks.yaml` - Central configuration
- `src/pipeline.py` - Main training pipeline
- `src/evaluation/evaluate.py` - Global evaluator
- `src/utils/config_loader.py` - YAML loader
- `src/utils/logger.py` - Logging utilities
- Model stubs: `hybrid_svm_knn.py`, `lstm_url.py`, `rnn_gru.py`, `webphish_cnn.py`, `egso_cnn.py`
- Feature stubs: `manual.py`, `sequential.py`, `rnn_gru.py`, `multimodal.py`, `tfidf_svd.py`
- Tests: `test_model_factory.py`, `test_feature_processors.py`, `test_evaluation.py`

## Next Steps

1. Implement model logic in `src/models/`
2. Implement feature processing in `src/features/`
3. Prepare and load datasets
4. Run training pipeline
5. Review comparison reports

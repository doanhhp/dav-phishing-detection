# Phishing Detection Benchmark Framework

A comparative evaluation of five phishing detection methods for malicious URL and HTML classification.

---

## Models

| Model | Type | Approach |
|---|---|---|
| **Hybrid SVM+KNN** | Traditional ML | Manually engineered features |
| **LSTM URL-Only** | Deep Learning | Raw character sequences of URLs |
| **WebPhish CNN** | Deep Learning | State-of-the-art multi-modal CNN (99.03% accuracy) |
| **EGSO-CNN (2025)** | Deep Learning | Optimized CNN utilizing both URL and HTML with TF-IDF and dimensionality reduction |
| **RNN-GRU** | Deep Learning | Sequential model for textual web data |

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure models**

Edit `config/benchmarks.yaml` to specify dataset paths, global settings, and model-specific hyperparameters.

**3. Run experiments**

Train and evaluate a single model:
```bash
python -m src.pipeline hybrid_svm_knn config/benchmarks.yaml
```

Or run the full benchmark suite across all five models:
```bash
for model in hybrid_svm_knn lstm_url webphish_cnn egso_cnn rnn_gru; do
    python -m src.pipeline $model config/benchmarks.yaml
done
```

**4. Generate comparison report**

Once models are trained, use the global evaluator to generate visual comparisons and a leaderboard:
```bash
python -m src.evaluation.evaluate experiments/ reports/comparison/
```

---

## Project Structure

```
PhishingDetection/
├── .gitignore                   # Ignored files (data, artifacts, pycache)
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
│
├── experiments/                 # Saved model weights (.pkl, .h5, .pt) and metrics, predictions
├── config/
│   └── benchmarks.yaml          # Master configuration file
├── data/
│   ├── raw/                     # Immutable original data (URL.xlsx, html.xlsx)
│   └── processed/               # Cleaned, engineered feature matrices
├── docs/                        # Detailed model references and integration summaries
├── notebooks/                   # Jupyter notebooks for EDA
├── scripts/
│   └── setup_benchmark.py       # Executable initialization scripts
│
├── src/                         # Core source code package
│   ├── evaluation/              # Metrics and visualization generators
│   ├── features/                # Feature processors (TF-IDF, Multimodal, Sequential)
│   ├── models/                  # Model architecture implementations
│   ├── utils/                   # Loggers and config loaders
│   └── pipeline.py              # Master training and evaluation pipeline
│
└── tests/                       # Unit tests for factory, features, and evaluation
```

---

## Model Integration

The framework uses a strict **Factory Pattern**. Each model requires three components:

| Component | Location |
|---|---|
| Architecture | `src/models/{model_name}.py` |
| Feature Processor | `src/features/{processor_type}.py` |
| Output Directory | Auto-generated at `experiments/{model_name}/` |

Implement custom model logic inside the stub files — the factory pattern handles instantiation, training loops, and metric logging.

---

## Evaluation Outputs

After running the global evaluator, the following are written to `reports/comparison/`:

| File | Description |
|---|---|
| `leaderboard.csv` | Comprehensive performance comparison table |
| `roc_curves.png` | Overlaid ROC curves for all models |
| `confusion_matrices.png` | Grid plot of all confusion matrices |
| `metrics_heatmap.png` | Heatmap of Accuracy, Precision, Recall, and F1-Score |

---

## Testing

Run the full test suite to verify pipeline integrity:
```bash
pytest tests/ -v
```

---

## Framework Features

**Modular** — Strict Factory pattern for seamless swapping of models and feature processors.

**Configurable** — A single YAML file controls all experiment settings.

**Extensible** — Add new architectures by dropping a script into `src/models/`.

**Production-Ready** — Adheres to MLOps and Data Science repository standards.
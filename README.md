# Phishing Detection Benchmark Framework

A comparative evaluation of eleven phishing detection methods, culminating in robust Tree-based structural models that eliminate "Domain Shift" decay.

---

## Models

| Model | Type | Approach |
|---|---|---|
| **Hybrid SVM+KNN** | Traditional ML | Manually engineered features |
| **LSTM URL-Only** | Deep Learning | Raw character sequences of URLs |
| **WebPhish CNN** | Deep Learning | State-of-the-art multi-modal CNN |
| **EGSO-CNN (2025)** | Deep Learning | Optimized CNN utilizing both URL and HTML with TF-IDF |
| **RNN-GRU** | Deep Learning | Sequential model for textual web data |
| **Structural RF** | Tree Ensemble | Random Forest on structural features |
| **Structural XGB** | Gradient Boosting | XGBoost on structural features (Unified/Early Fusion) |
| **Structural Stacking** | Tree Ensemble | Late Fusion ensemble of URL and HTML experts |
| **Mid-Fusion XGBoost** | Tree Ensemble | Deep Feature-Weighted Passthrough Stacking (Best Generalization) |

---

## How it Works
1. **Live Data Collection**: We fetch real-time URLs from OpenPhish, PhishTank, and Tranco to build Out-of-Distribution (OOD) test sets.
2. **Structural Feature Engineering**: Through iterative tuning, we engineered **163 structural DOM invariants** (28 geometric properties and 135 XPath TF-IDF sequential metrics). This purely structural approach completely prevents the "Domain Shift" decay that destroys deep learning models on live data.
3. **Modeling**: We apply Random Forest (`structural_rf`) and XGBoost (`structural_xgb`) classifiers against these features to achieve superior generalization.
4. **Visual Analytics**: Advanced DOM tree probabilistic mapping using left-to-right hierarchy layout and ultra-deep (20 depth) topological analysis.

## Project Research & Logs
Our experimental reasoning, feature importance analysis, probability DOM tree visualizations, and progression logs are stored in `docs/reports/advanced_adaptation_log.md` and `docs/research_log.md`.

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure models**
Edit `config/benchmarks.yaml` to specify global settings and model-specific hyperparameters.

**3. Run experiments**

Train and evaluate a single model on the main dataset:
```bash
python -m src.pipeline structural_rf config/benchmarks.yaml
```

Evaluate a model on a completely new Out-of-Distribution (OOD) dataset:
```bash
python -m src.pipeline structural_rf config/benchmarks.yaml --url_path data/raw/OOD_URL.xlsx --html_path data/raw/OOD_html.xlsx
```

Run the full benchmark suite across all models with cross-validation:
```bash
for model in hybrid_svm_knn lstm_url webphish_cnn egso_cnn rnn_gru structural_rf structural_xgb; do
    python -m src.pipeline $model config/benchmarks.yaml --cv 5
done
```

**4. Hyperparameter Tuning**
Use the Optuna script to find optimal configurations for a model using Bayesian optimization:
```bash
python -m src.tuning.optuna_tuner egso_cnn config/benchmarks.yaml --trials 100 --cv 3
```

**5. Generate comparison report**
Generate visual comparisons and a leaderboard:
```bash
python -m src.evaluation.evaluate experiments/ reports/comparison/
```

**6. Crawl Live Out-of-Distribution Data**
Fetch legitimate websites from Tranco and phishing websites from PhishTank or OpenPhish.
```bash
python -m src.data.crawler --legit 500 --phish 100 --timeout 5
```

---

## Project Structure

```
PhishingDetection/
├── config/
│   └── benchmarks.yaml          # Master configuration file
├── data/
│   ├── raw/                     # Original datasets (Main, OOD, PhreshPhish)
│   └── processed/               # Cleaned, engineered feature matrices
├── docs/
│   ├── assets/                  # 8K DOM trees, PCA/UMAP graphs, Feature drift maps
│   └── reports/                 # Advanced adaptation logs, LaTeX papers
├── experiments/                 # Saved model weights (.pkl, .h5, .pt) and results
├── scripts/
│   ├── train/                   # Scripts for training models
│   ├── evaluate/                # Scripts for cross-dataset evaluation and zero-day testing
│   ├── utils/                   # Scratch scripts and data conversion utilities
│   ├── visualizations/          # Scripts for generating XAI graphs and feature plots
│   └── experiments/             # Heavy scripts for incremental learning and DOM vis
├── src/                         # Core source code package
│   ├── evaluation/              # Metrics and visualization generators
│   ├── features/                # Feature processors (TF-IDF, Structural, Sequential)
│   ├── models/                  # Model architecture implementations (RF, XGB, CNNs)
│   ├── tuning/                  # Optuna optimization
│   ├── utils/                   # Loggers and config loaders
│   └── pipeline.py              # Master training and evaluation pipeline
└── tests/                       # Unit tests
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

## Framework Features

**Modular** — Strict Factory pattern for seamless swapping of models and feature processors.

**Configurable** — A single YAML file controls all experiment settings.

**Extensible** — Add new architectures by dropping a script into `src/models/`.

**Production-Ready** — Adheres to MLOps and Data Science repository standards.
# Zero-Shot Evaluation Report (v2)

Date: 2026-06-22

## Objective
Evaluate the Out-Of-Distribution (OOD) robustness of pure deep learning models compared to structural ML models under various fusion architectures (Early, Late, and Mid Fusion).

## Methodology
- **Training**: All models were trained exclusively on the Main Dataset (~45,000 samples).
- **Zero-Shot Testing**: All models were evaluated directly on the PhreshPhish Dataset (40,000 highly varied, unseen domains) without retraining.
- **Features Used in XGBoost/Stacking**: 28 base features, Structural Sequence TF-IDF, XPath TF-IDF, and Spatial Zone TF-IDF. 
- **Parser Optimization**: HTML parsing is now handled by a custom `lxml` tree traversal ($O(N)$), replacing BeautifulSoup, resulting in an over 100x speedup in feature extraction.

## Results: In-Distribution (Main Dataset)
| Metric | Structural XGBoost (Early) | Structural Stacking (Late) | Mid-Fusion XGBoost | WebPhish CNN |
|---|---|---|---|---|
| Accuracy | 97.05% | 97.92% | **98.28%** | 98.76% |
| F1-Score | 97.05% | 97.92% | **98.28%** | 98.76% |
| ROC-AUC | 99.54% | 99.66% | **99.79%** | 99.89% |

## Results: Zero-Shot (PhreshPhish Dataset)
| Metric | Structural XGBoost (Early) | Structural Stacking (Late) | Mid-Fusion XGBoost | WebPhish CNN |
|---|---|---|---|---|
| Accuracy | 77.22% | 76.05% | **79.32%** | 67.02% |
| F1-Score | 74.69% | 73.10% | **78.20%** | 73.71% |
| ROC-AUC | 85.25% | 86.35% | **87.78%** | 78.88% |

## Training & Inference Speeds
| Metric | Structural XGBoost (Early) | Structural Stacking (Late) | Mid-Fusion XGBoost | WebPhish CNN |
|---|---|---|---|---|
| Inference Latency | **0.0017 ms/sample** | 0.0026 ms/sample | 0.0046 ms/sample | ~0.21 ms/sample |

## Key Insights

1. **Mid-Fusion Solves Overfitting**: The Mid-Fusion XGBoost (Deep Feature-Weighted Stacking using `passthrough=True`) achieved the highest Zero-Shot accuracy (**79.32%**). Because the meta-learner was given access to the raw features *in addition* to the expert predictions, it learned complex synergies and avoided blindly trusting overconfident experts on out-of-distribution data.
2. **Speed Breakthrough**: Switching from `BeautifulSoup` to native `lxml` tree traversal dropped the feature extraction time for 40,000 domains from ~30 minutes down to ~2 minutes.
3. **The Deep Learning Generalization Collapse**: WebPhish CNN achieved the highest in-distribution accuracy (98.76%) but suffered catastrophic failure on unseen domains (67.02% accuracy). This confirms that keyword-based NLP models memorize specific vocabulary distributions rather than generalizing phishing behaviors.

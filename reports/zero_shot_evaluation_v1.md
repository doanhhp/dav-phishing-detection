# Zero-Shot Evaluation Report (v1)

Date: 2026-06-22

## Objective
Evaluate the Out-Of-Distribution (OOD) robustness of pure deep learning models (WebPhish CNN) compared to unified structural ML models (Structural XGBoost with Spatial Zones) and Late Fusion Stacking architectures.

## Methodology
- **Training**: All models were trained exclusively on the Main Dataset (~4,000 samples).
- **Zero-Shot Testing**: All models were evaluated directly on the PhreshPhish Dataset (40,000 highly varied, unseen domains) without retraining.
- **Features Used in XGBoost/Stacking**: 28 base features, Structural Sequence TF-IDF, XPath TF-IDF, and Spatial Zone TF-IDF.

## Results: In-Distribution (Main Dataset)
| Metric | Unified Structural XGBoost | Structural Stacking (Late Fusion) | WebPhish CNN |
|---|---|---|---|
| Accuracy | 97.05% | **97.92%** | 98.76% |
| F1-Score | 97.05% | **97.92%** | 98.76% |
| ROC-AUC | 99.54% | **99.66%** | 99.89% |

## Results: Zero-Shot (PhreshPhish Dataset)
| Metric | Unified Structural XGBoost | Structural Stacking (Late Fusion) | WebPhish CNN |
|---|---|---|---|
| Accuracy | **77.22%** | 76.05% | 67.02% |
| F1-Score | **74.69%** | 73.10% | 73.71% |
| ROC-AUC | 85.25% | **86.35%** | 78.88% |

## Training & Inference Speeds
| Metric | Unified Structural XGBoost | Structural Stacking (Late Fusion) | WebPhish CNN |
|---|---|---|---|
| Training Time (Main) | ~5 seconds | ~10 seconds | ~6 minutes |
| Inference Latency | **0.0017 ms/sample** | 0.0026 ms/sample | ~0.21 ms/sample |

## Key Insights

1. **The Deep Learning Generalization Collapse**: WebPhish CNN achieved the highest in-distribution accuracy (98.76%) but suffered catastrophic failure on unseen domains (67.02% accuracy). This confirms that keyword-based NLP models memorize specific vocabulary distributions rather than generalizing phishing behaviors.
2. **Unified Synergy Outperforms Stacking**: Interestingly, while the Late Fusion Stacking model improved in-distribution performance (from 97.05% to 97.92%), it actually dropped in Zero-Shot performance (77.22% down to 76.05%). The Unified Structural XGBoost generalized better because it was able to learn cross-modal synergistic interactions (e.g., combining URL anomalies directly with Spatial Zone HTML structures in the same tree), whereas the Stacking model isolated them and became more prone to overfitting the specific expert boundaries.

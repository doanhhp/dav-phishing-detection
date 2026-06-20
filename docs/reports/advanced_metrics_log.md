# Advanced Metrics Comparison

This report details the operational performance metrics of the Pruned XGBoost model against the WebPhish CNN on the 40k PhreshPhish dataset.

| Metric | Pruned XGBoost | WebPhish CNN |
|---|---|---|
| Accuracy | 0.9691 | 0.9729 |
| F1-Score | 0.9691 | 0.9727 |
| F2-Score | **0.9677** | 0.9700 |
| ROC-AUC | 0.9950 | 0.9959 |
| False Positive Rate (FPR) | **0.0285** | 0.0225 |
| False Negative Rate (FNR) | 0.0333 | 0.0318 |
| Inference Latency (ms/sample)| **0.0011 ms** | 0.1897 ms |

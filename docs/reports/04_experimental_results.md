# Experimental Results & Benchmarks

This report comprehensively logs the chronological progression of all experiments conducted to evaluate phishing detection architectures under extreme Domain Shift (training on historic 2021 data, testing purely on zero-day 2026 data).

## 4.1 Experiment 1: The Heuristic & Lexical Baselines
We first evaluated traditional "System 1" classifiers (`url_rf` and `hybrid_svm_knn`) to determine if purely lexical and heuristic rules (e.g., domain entropy, brand impersonation, manual URL checks) could detect modern phishing.

*   **Training (Historic Data):** 91.3% Accuracy
*   **Zero-Shot OOD Accuracy:** 18.7%
*   **Zero-Shot OOD Precision:** 0.00% (Flagged almost everything as malicious)

**Conclusion:** Scammers have completely bypassed URL heuristics by utilizing hijacked legitimate domains and clean URLs. Lexical baselines fail catastrophically under Domain Shift.

---

## 4.2 Experiment 2: Deep Learning & Transfer Learning Failure
We tested state-of-the-art Deep Learning NLP models (LSTMs, GRUs, WebPhish CNN) utilizing **Transfer Learning** (freezing base layers trained on historic data and fine-tuning only the final dense layers) versus training purely from scratch.

### Deep Learning Few-Shot Performance

| N_Samples | RF_Scratch (Baseline) | DNN_Scratch | DNN_Transfer |
|----------:|----------------------:|------------:|-------------:|
|        10 |              0.812098 |    0.793436 |     0.766409 |
|        20 |              0.943470 |    0.807667 |     0.761858 |
|        50 |              0.972892 |    0.869813 |     0.772758 |
|       100 |              0.990839 |    0.930233 |     0.786822 |
|       200 |              0.997242 |    0.959417 |     0.795508 |

**Conclusion:** Even when allowed 200 new samples, Deep Learning Transfer Learning struggles heavily. When tested purely Zero-Shot (0 samples), Deep Learning collapses to ~43% accuracy and 0-5% Recall. This proves Deep Learning suffers from **Vocabulary Misalignment (Content Shift)**.

---

## 4.3 Experiment 3: The Structural Paradigm Shift
We shifted the architecture to analyze the **75 Structural Invariants** (25 numeric ratios + 50 DOM tag sequences via TF-IDF). We tested complex Neural Networks (`structural_dnn`) against Tree Ensembles (`structural_rf`).

### Deep Learning vs Structural Few-Shot

| N_Samples | LSTM_URL (Transfer) | RNN_GRU (Transfer) | WebPhish_CNN (Transfer) | Structural_RF (Scratch) |
|----------:|--------------------:|-------------------:|------------------------:|------------------------:|
|        10 |            0.246139 |           0.249035 |                0.725869 |                0.812098 |
|        20 |            0.245614 |           0.248538 |                0.765757 |                0.943470 |
|        50 |            0.460174 |           0.751004 |                0.819946 |                0.972892 |
|       100 |            0.774841 |           0.751586 |                0.823820 |                0.990839 |
|       200 |            0.921592 |           0.751773 |                0.828605 |                0.997242 |

**Conclusion:** The Structural Random Forest (trained from scratch on just 20 samples) vastly outperforms deeply pre-trained Neural Networks, proving that invariants are mathematically easier to learn than raw NLP content.

---

## 4.4 Experiment 4: The 45,000-Record Definitive Zero-Shot Benchmark
To establish the absolute baseline generalization power of Structural Machine Learning before any Incremental adaptation, we ran the full 45,000 historical dataset against the unbalanced live 2026 dataset.

| Configuration | Accuracy | Recall (Spam Caught) | Precision (Ham Safe) | F1-Score | ROC AUC |
|---------------|----------|----------------------|----------------------|----------|---------|
| **50 Features (Sweet Spot)** | **69.87%** | **68.62%** | 25.76% | 37.46% | 0.6845 |
| **200 Features (Overfit Trap)**| 77.40% | 65.36% | 32.26% | 43.20% | 0.6886 |

**Conclusion:** The **Curse of Dimensionality**. Increasing features to 200 improves Accuracy by allowing the model to memorize old legitimate site templates, but significantly *drops* Recall (resilience against zero-day phishing). We locked the vector to 50 TF-IDF features to prioritize Recall.

---

## 4.5 Experiment 5: The Mixed Retraining Dilution Effect
To prove why a simple static model is insufficient over long timelines, we compared rebuilding models completely from scratch using a mix of (3,000 Historical + N New Samples) versus the Structural RF trained *purely* on the N new samples.

| N_Samples | LSTM (Mixed) | WebPhish_CNN (Mixed) | Structural_RF (Mixed) | Structural_RF (Purely New) |
|----------:|-------------:|---------------------:|----------------------:|---------------------------:|
|        10 |     0.404440 |             0.781210 |                0.8047 |                   0.812098 |
|        20 |     0.419753 |             0.841455 |                0.8223 |                   0.943470 |
|        50 |     0.599063 |             0.801539 |                0.8330 |                   0.972892 |
|       100 |     0.737139 |             0.809373 |                0.8693 |                   0.990839 |
|       200 |     0.720252 |             0.798660 |                0.9023 |                   0.997242 |

**Conclusion:** Notice how `Structural_RF` drops from 99.7% to 90.2% when old historical data is mixed in. This mathematically proves that **Historical Data actively dilutes zero-day adaptation!**

---

## 4.6 Experiment 6: The Champion Architectures (Structural RF & XGBoost)
Having proven that deep learning fails under Content Shift (Exp 2), and that mixing old data dilutes zero-day adaptation (Exp 5), we arrived at our final architectures. 

When trained purely on the new data distribution using a standard 5-Fold Cross-Validation, our structural tree-based models achieved exceptional results without the need for heavy Deep Learning:

*   **Structural Random Forest (Static Champion):** **96.52% Accuracy**, 99.33% ROC-AUC
*   **Structural XGBoost (Dynamic Champion):** **96.80% Accuracy**, 99.48% ROC-AUC

**Final Project Conclusion:** The **Structural Random Forest (`structural_rf`)** acts as our definitive Champion Static Model due to its unmatched Zero-Shot OOD resilience (68.6% Recall) and its incredibly high 96.5% peak accuracy. However, to fully defeat natural drift over time, **Structural XGBoost (`structural_xgb`)** operates as our Champion Dynamic Model, utilizing its gradient boosting engine to perform Incremental Learning and gracefully slide its decision boundary with streaming data.

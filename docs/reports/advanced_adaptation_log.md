# Advanced Domain Adaptation & Retraining Log

This document chronicles the final operational tests conducted to mathematically prove the limits of Incremental Learning and Retraining Strategies across our phishing datasets.

## Experiment 1: The Limits of Incremental Learning (The 32k Plateau)

**Hypothesis:** Can incremental learning completely replace retraining from scratch if a model is exposed to a vastly more complex dataset?
**Setup:** 
- Base Model trained on the simple, historical 11k dataset.
- Target Domain: The highly sophisticated 40k PhreshPhish dataset.
- We streamed 32,000 PhreshPhish samples into the simple base model to see if it could eventually reach 95%+ accuracy.

**Results:**
| PhreshPhish Samples Streamed | Holdout Test Accuracy |
|------------------------------|-----------------------|
| **0 (Frozen Baseline)**      | 75.94%                |
| **1,000**                    | 90.48%                |
| **5,000**                    | 91.31%                |
| **15,000**                   | 91.50%                |
| **32,000 (Exhausted)**       | 90.30%                |

**Conclusion (Architectural Expansion Failure):** 
Incremental learning hit a mathematical brick wall at ~92% and ultimately plateaued around 90.30%. Because the base trees from the 11k dataset were too shallow/simple, the incremental gradient updates could only add superficial correction trees. It proved that Incremental Learning is perfect for threshold drift (Covariate Shift) but **cannot invent fundamentally missing complex architectural branches**. To cross 95%, a full retrain from scratch is strictly required.

---

## Experiment 2: Retraining Speed (XGBoost vs Random Forest)

**Hypothesis:** If forced to fully retrain from scratch due to massive architectural shifts, which tree-based algorithm is operationally superior?
**Setup:** Train both models on the full 40k PhreshPhish dataset from scratch using all CPU cores.

**Results:**
- **Random Forest:** 5.92 seconds (96.53% Accuracy)
- **XGBoost:** 8.67 seconds (96.91% Accuracy)

**Conclusion:**
Random Forest is **1.5x FASTER** to retrain due to its highly parallel Bagging architecture (all 250 trees build simultaneously). XGBoost's sequential Boosting nature makes it slightly slower but edges out a **0.39% higher accuracy**. 

---

## Experiment 3: The Data Dilution Effect (Few-Shot Retraining)

**Hypothesis:** If we must retrain Random Forest from scratch to adapt to new phishing templates, can we just mix a few new samples into the old dataset?
**Setup:** 
- Base Pool: 45,000 historical samples.
- Injection Pool: 0 to 5,000 target-domain samples from PhreshPhish.
- Fully retrain Random Forest on the mixed dataset and evaluate on the target domain holdout.

**Results:**
| Target Samples Injected | Accuracy on Target Domain |
|-------------------------|---------------------------|
| **0**                   | 73.12%                    |
| **500**                 | 84.69%                    |
| **2,000**               | 89.41%                    |
| **5,000**               | 92.19%                    |

**Conclusion (Data Dilution):**
Even with 5,000 new samples injected, Random Forest only achieved 92.19% (failing to reach its theoretical 96.53% maximum). Because Bagging assigns equal weight to all samples, the 45,000 old historical templates mathematically "shouted down" the 5,000 new templates during tree construction. 

**Ultimate Pipeline Verdict:** 
To adapt to a massive domain shift, simply dumping new data into an old dataset creates dilution. A security pipeline must either:
1. Use **XGBoost Incremental Learning** (which sequentially forces trees to focus exclusively on the new data without old data drowning it out).
2. Use **Random Forest Full Retraining**, but aggressively prune/throw away old data so the trees can freely mold to modern structures.

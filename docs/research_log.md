# Phishing Detection Research Log

## 1. Initial State
The project began by evaluating the `webphish_cnn` architecture (accuracy: 65%) and a basic `lstm_url` model. Neural network architectures heavily rely on textual tokens (words in the URL and HTML).

## 2. The Domain Shift Problem
When tested on **Out-of-Distribution (OOD)** live data from the internet, models trained on static lexical features suffered massive performance degradation. 
- **Cause:** Attackers dynamically change the tokens (e.g., using random domains, obfuscating HTML) to easily bypass these filters.
- **Our Hypothesis:** Mathematical structural features (length, element counts, ratios) are *invariant* to these obfuscation tactics.

## 3. Structural Feature Engineering
We built `StructuralProcessor` to extract numerical properties rather than raw text. 
Initially, we extracted 15 features. We then expanded this to **25 features** to deeply analyze structural anomalies:
- Path depths
- HTTPS usage
- Mailto links
- Password fields
- Obfuscation techniques (`unescape`, `eval`)
- Empty anchor tags

## 4. Modeling & Results
We applied Tree-based models (Random Forest, XGBoost) to this structural tabular data, as tree-based models significantly outperform Deep Learning on small-dimension tabular datasets.

### Feature Importances (Random Forest)
After plotting the Gini Importances, our Top 5 Structural Features were:
1. `html_length` (0.245)
2. `html_empty_links` (0.156) - A huge indicator, as phishing kits often have dead links in their visual clones!
3. `html_num_tags` (0.105)
4. `html_script_count` (0.063)
5. `html_title_length` (0.055)

### OOD Leaderboard
By running `evaluate_ood.py`, we proved our hypothesis:
- `structural_rf` achieved **71.7% Accuracy** and **0.744 ROC AUC** on the live internet data.
- This successfully outperformed the `webphish_cnn` baseline (65.3%), proving that structural invariants are much more resistant to Domain Shift than textual tokens.

## Next Steps
To continue improving OOD robustness without active retraining, future research should focus on:
1. Extracting external metadata constraints (WHOIS domain age, TLS certificate validation).
2. Screenshot-based visual similarity analysis (CNNs on rendered web pages).

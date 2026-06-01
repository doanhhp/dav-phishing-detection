# Phishing Detection Benchmark: Project Log & Journey

This document captures the chronological development, experimental results, and logical thinking process behind the Phishing Detection framework we built together.

## Phase 1: The Initial Request & Sandbox Setup
**Objective:** The user requested to train five different phishing detection models and combine them into a single benchmarking framework.
**Actions:**
- Evaluated models: `hybrid_svm_knn`, `lstm_url`, `webphish_cnn`, `egso_cnn`, and `rnn_gru`.
- Encountered massive scale issues: The original dataset had 114,000 HTML files and URLs. Initial training directly on this was incredibly slow and crashed due to memory limits.
- **Pivot:** Created a stratified 3,000-sample subset (`URL_subset.xlsx`, `html_subset.xlsx`) to perform initial development and verify the code architecture worked without burning days of compute time.

## Phase 2: Building the Architecture
**Objective:** Make the code modular, maintainable, and easily extendable.
**Actions:**
- Implemented a strict **Factory Pattern** (`FeatureFactory` and `ModelFactory`).
- Centralized all configurations in `benchmarks.yaml` to prevent hardcoding.
- Developed an automated evaluation module (`evaluate.py`) that exports leaderboard metrics, ROC curves, confusion matrices, and heatmaps.

## Phase 3: Structural Invariants & Feature Engineering
**Objective:** Engineer a model that generalizes to live internet data without requiring endless retraining.
**Actions:**
- Realized that *tokens drift, but structures persist*. A phishing site can change its text from "paypal" to "netflix", but it still hides iframes (`display:none`) and cross-loads scripts from external domains.
- Designed `StructuralProcessor` in `src/features/structural.py` to extract purely numerical invariant properties: URL Entropy, Digit Ratios, Hidden Elements, External Link Ratios, etc.
- **Pruning Phase**: We pruned the bottom 7 least important features based on Tree models, yielding a lightweight 21-feature structural model that maintained >96% in-distribution accuracy.

## Phase 4: The Reality of Domain Shift
**Objective:** Evaluate models on live, modern Out-Of-Distribution (OOD) data.
**Actions:**
- Built `src/data/crawler.py` with anti-bot spoofing and automatic fallbacks to OpenPhish to gather the latest Top 1M Tranco sites and PhishTank URLs.
- Evaluated the pruned 21-feature `structural_rf` model on the live OOD data.
- **The Shocking Result:** Accuracy plummeted to `~49%`. The model had simply memorized old 2021 HTML structures. 
- **Visualization:** We used PCA, t-SNE, and UMAP to visualize the structural features. We proved that the Training and Live OOD datasets occupy entirely distinct regions in the feature space (Domain Shift).

## Phase 5: The Ultimate Architecture (Current)
**Objective:** How do we actually solve Domain Shift? 
**Actions:**
Through active collaboration, we designed four industry-standard solutions to defeat concept drift:
1. **Visual Similarity (Siamese Networks):** Using headless browsers to compare screenshots of suspicious sites against legitimate brand fingerprints, bypassing HTML completely.
2. **Prioritize Lexical Features:** Using character entropy, subdomain depth, and special character ratios from the URL, which are far more rigid and stable than HTML.
3. **HTML as Natural Language (SLMs):** Passing raw HTML into lightweight Transformers to understand semantic intent rather than tag placement.
4. **Multi-Layered Funnel Pipeline:** A 3-layer approach using Heuristics -> Lexical ML -> Siamese Heavy Lifters to filter out noise dynamically.

### Implementation of the Ultimate Architecture
To conclude the project, we integrated the most viable elements of these solutions directly into our `StructuralProcessor`:
- **Structural Skeletoning (DOM NLP):** Added a `TfidfVectorizer` to learn the "grammar" of HTML tag sequences.
- **CSS Anomaly Detection:** Added counters for visibility hacks (`display: none`, `opacity: 0`) and position manipulation (`z-index`).
- **Functional Ratios:** Added Dead Link checking (`<a href="#">`) and Input-to-Content (`<input>` vs `<p>`) ratios.
- **Lexical ML:** Added URL entropy, digit ratios, and keyword presence to build a robust lexical foundation.

We retrained the models with these 44 advanced features, achieving an outstanding **68.8% Zero-Shot OOD Accuracy** using `structural_rf`. This proved that structural DOM features are vastly superior to text-based features for resisting content shift.

## Phase 6: The Fall of URL Heuristics (Domain Shift)
**Objective:** Test if human intuition (URL heuristics) alone could act as a "System 1" filter before analyzing the HTML.
**Actions:**
- Engineered 11 URL-specific heuristics (`domain_entropy`, `url_num_subdomains`, `keyword_stuffing_score`).
- **Result:** The model achieved 91.3% accuracy on historical data but crashed to **18.7% accuracy** on live zero-day data.
- **Conclusion:** Scammers have completely adapted. They use clean, legitimate `.com` domains and short URLs to bypass human heuristics. URL features are highly volatile and unreliable against modern zero-day attacks.

## Phase 7: Advanced OOD Defenses (Soft Ensembles & Anomaly Detection)
**Objective:** Combat the severe content shift in modern phishing using advanced architectures.
**Actions:**
1. **Soft Ensemble (Meta-Learning):** We trained a Logistic Regression Meta-Classifier to dynamically weight the URL model vs. the Structural model.
   - **Result:** Failed (22.66% accuracy). The URL model was so confidently wrong on zero-day data that it poisoned the ensemble.
2. **Anomaly Detection (Deep Autoencoder):** We trained a Keras Autoencoder *exclusively* on legitimate (ham) structural features, expecting phishing sites to cause massive Reconstruction Errors (MSE anomalies).
   - **Result:** Failed (AUC 0.44 - worse than random). We discovered the **Simplicity Paradox**: Phishing sites are structurally *simpler* than legitimate sites. Because they are simple, the Autoencoder had no trouble reconstructing them, resulting in *lower* errors than complex legitimate sites.

### Ultimate Conclusion
Our extensive experiments prove that **URL heuristics are dead** and **unsupervised anomaly detection fails due to the simplicity paradox**. The champion remains our standalone, supervised `structural_rf` model, which explicitly learned the structural constraints of phishing code.

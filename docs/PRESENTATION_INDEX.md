# Presentation Guide: Defeating Phishing Content Shift

This document organizes the key graphs and analyses into a logical narrative flow for your presentation.

## Part 1: The Problem (Domain Shift)
*Prove to the audience that 2021 data is vastly different from 2026 data.*

1. **[Dimensionality Reduction (t-SNE)](assets/domain_shift/domain_shift_t-sne.png)**
   - **Key Point:** The 2021 training data (red) and 2026 zero-day data (blue) occupy completely different spaces. This is why models trained in 2021 fail in 2026.
2. **[Evolution of Phishing URLs](assets/domain_shift/side_by_side_url.png)**
   - **Key Point:** Comparing 2021 vs 2026 shows that scammers completely changed their URL structures (length and entropy).
3. **[Structural Shift](assets/domain_shift/side_by_side_structure.png)**
   - **Key Point:** The underlying HTML structure (DOM depth, Tag Diversity) of phishing sites has also shifted over 5 years.

## Part 2: The Modern Threat (Zero-Day Analysis)
*What does a modern 2026 phishing attack actually look like compared to a legitimate site?*

1. **[Structural Complexity (The Simplicity Paradox)](assets/zero_day_analysis/ood_eda_structure.png)**
   - **Key Point:** Legitimate websites are incredibly complex. Modern zero-day phishing sites are structurally stripped down.
2. **[Modern Evasion Tactics](assets/zero_day_analysis/ood_eda_evasion.png)**
   - **Key Point:** Modern phishing heavily relies on iframes and hidden elements (`display:none`) to deceive users and evade scrapers.

## Part 3: The Solution (Structural Features)
*Explain how your Random Forest focuses on structure instead of text.*

1. **[Feature Importance](assets/feature_importance/feature_importance.png)**
   - **Key Point:** The model ignores words and heavily weights structural indicators (external resources, DOM depth, and hidden elements).

## Part 4: The Ultimate Proof (Model Comparisons)
*Prove that your Structural Random Forest outperforms Deep Learning models.*

1. **[Transfer Learning Failure](assets/model_comparisons/dl_few_shot_comparison.png)**
   - **Key Point:** Trying to fine-tune a pre-trained Deep Learning model on the new data fails because the frozen text embeddings are anchored to 2021 keywords (e.g., "paypal" instead of "crypto").
2. **[Retraining from Scratch Failure (Catastrophic Dilution)](assets/model_comparisons/dl_retrain_comparison.png)**
   - **Key Point:** Even if you completely rebuild the Deep Learning model from scratch by mixing 3,000 old samples with 200 new samples, the new zero-day keywords are treated as statistical noise. The old data actively dilutes the new data.
3. **[The Winning Strategy](assets/model_comparisons/few_shot_comparison.png)**
   - **Key Point:** The absolute best strategy is to throw away the historical data entirely and train a lightweight Structural Random Forest purely on a handful of new samples. It adapts instantly to structural invariants and hits 99.7% accuracy.

---
**Detailed Reports:**
For detailed statistical breakdowns, view the reports in the `reports/` folder.
*   [Architecture and Models](reports/architecture_and_models.md)
*   [Exploratory Data Analysis](reports/exploratory_data_analysis.md)
*   [Experimental Results](reports/experimental_results.md)
*   [Project Logs and History](reports/project_logs_and_history.md)

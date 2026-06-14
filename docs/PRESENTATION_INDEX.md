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
3. **[The "Dead Link" Phenomenon](assets/zero_day_analysis/adv_eda_dead_links.png)**
   - **Key Point:** Phishing sites don't bother building "About Us" pages. An enormous percentage of their links point to nowhere (`href="#"`), exposing their laziness.
4. **[Text-to-Code Ratio](assets/zero_day_analysis/adv_eda_text_to_code.png)**
   - **Key Point:** Phishing sites are mostly HTML code (forms and images) with very little actual readable paragraph text compared to legitimate sites.
5. **[Input Tag Density](assets/zero_day_analysis/adv_eda_input_density.png)**
   - **Key Point:** Scammers have a hyper-focus on stealing data, resulting in a significantly higher density of `<input>` fields per page.
6. **[URL Obfuscation](assets/zero_day_analysis/adv_eda_url_obfuscation.png)**
   - **Key Point:** Scammers hide behind deeply nested paths (`/login/secure/step1`) and high subdomain counts (`auth.update.paypal.xyz`) to trick users.

## Part 3: The Solution (Structural Features)
*Explain how your Random Forest focuses on structure instead of text.*

1. **[Feature Importance](assets/feature_importance/feature_importance.png)**
   - **Key Point:** The model ignores words and heavily weights structural indicators (external resources, DOM depth, and hidden elements).

## Part 4: Explainable AI & Structural Invariants
- **[Explainable AI Report](file:///d:/Desktop/PhishingDetection/docs/reports/explainable_ai_report.md)**
  - Why text-based models are "black boxes".
  - Proving the "Simplicity Paradox" via SHAP visualization (DOM Depth & Tag Diversity).
  - Unmasking Zero-Day templates using Form Action anomalies.

## Part 5: The Ultimate Architecture (Incremental XGBoost)
- **[Model Comparisons](file:///d:/Desktop/PhishingDetection/docs/assets/model_comparisons/)**
  - The Failure of Unsupervised Anomaly Detection (Autoencoders).
  - Why standard XGBoost suffers from Catastrophic Dilution on historical data.
  - The breakthrough of **Incremental Learning (Rolling Window)** achieving 94.1% Zero-Day Accuracy.

---
**Detailed Reports:**
For detailed statistical breakdowns, view the reports in the `reports/` folder.
*   [Architecture and Models](reports/architecture_and_models.md)
*   [Exploratory Data Analysis](reports/exploratory_data_analysis.md)
*   [Explainable AI Interpretability](reports/explainable_ai_report.md)
*   [Experimental Results](reports/experimental_results.md)
*   [Project Logs and History](reports/project_logs_and_history.md)

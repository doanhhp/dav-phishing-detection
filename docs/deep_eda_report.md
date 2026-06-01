# Deep Structural EDA Report

## The Core Problem: Why "Good" Models Fail in the Wild
In our previous experiments, Deep Learning models (CNNs, LSTMs) achieved >96% validation accuracy but plummeted to ~65% on the live Out-of-Distribution (OOD) dataset. 

This failure occurs because deep learning on text (TF-IDF, word embeddings) relies on the semantic meaning of the words in the HTML and URL. Phishing attackers constantly change these words, domains, and HTML structures specifically to bypass text-based models (Domain Shift).

To solve this, we must find the **mathematical structural signature** of a phishing kit—features that attackers *cannot* easily change without breaking their website.

## Findings from the Deep Structural EDA
We executed an advanced exploratory data analysis (available in `notebooks/deep_structural_eda.ipynb`) analyzing both the training and live OOD datasets. 

Here are the invariant structural characteristics of Phishing vs. Legitimate websites:

### 1. DOM Tree Complexity
Modern legitimate web applications (like PayPal, Microsoft, Chase) are incredibly complex, single-page applications built with React/Angular. They have deeply nested DOM trees (e.g., max depth > 20) and a high diversity of HTML tags.
- **Phishing Signature:** Phishing kits are often hastily assembled clones or single-page credential harvesters. They exhibit a much **shallower DOM tree** and a lower diversity of HTML tags. Attackers prioritize visual similarity, not structural complexity.

### 2. External Resource Dependency
To make a phishing site look exactly like the real brand, attackers need the brand's CSS, logos, and scripts. However, they usually host the HTML file on their own malicious server.
- **Phishing Signature:** Phishing sites display a remarkably high ratio of **external resources**. Their `<img src="...">` and `<link href="...">` tags almost entirely point to external domains (the legitimate brand's servers), whereas legitimate sites host a majority of their resources internally.

### 3. URL Lexical Topography (Entropy)
Attackers use fast-flux DNS, randomly generated domains (DGA), and highly complex subdomain chains (`paypal.update.secure-login-attempt.xyz`) to evade blocklists.
- **Phishing Signature:** The Shannon Entropy of phishing URLs is statistically higher than legitimate URLs. Random strings of characters or extremely long, hyphenated subdomain chains create a high-entropy signature that is rare in legitimate corporate domains.

## Conclusion & Next Steps
We have successfully identified the core underlying problem. The final model should **not** rely on TF-IDF or Word Embeddings.

Instead, the final architecture should be a **Tree-Based Ensemble (Random Forest or XGBoost)** trained exclusively on these deep structural invariant features. By ignoring *what* the page says, and focusing entirely on *how* it is mathematically structured, we create a classifier that is highly resistant to Domain Shift and evasion tactics!

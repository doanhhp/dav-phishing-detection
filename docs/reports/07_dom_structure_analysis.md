# Analysis & Insights: Probabilistic DOM Tree Visualizations

By visualizing the probabilistic DOM structures across the 2021 (`Main`), 2026 (`OOD`), and `PhreshPhish` datasets, several critical insights emerge regarding the fundamental geometric differences between legitimate web architecture and phishing campaigns.

These visualizations visually corroborate the mathematical findings from our SHAP analysis and Zero-Day evaluation.

---

## 1. The "Hollow Shell" Architecture of Phishing

Across all three datasets, the **Phishing DOM Trees** are consistently shallower, far less dense, and structurally simpler than their legitimate counterparts. 

- **Legitimate Sites:** Genuine, modern web applications (especially in the `OOD` and `Main` datasets) display massive, sprawling DOM trees. They require deeply nested components for navigation bars, footers, sidebars, interactive modules, and responsive grids. 
- **Phishing Sites:** Phishing templates are primarily designed for a single purpose: credential extraction. Their DOM trees resemble a "Hollow Shell." They consist of a basic structural wrapper (`html` > `body` > `div`), a few visual elements (`img`, `p`), and an immediate path to an `input` or `form` tag. They completely lack the structural "baggage" of a real web application.

## 2. Template Homogeneity (The Heatmap Phenomenon)

The continuous logarithmic color scaling (the heatmap) reveals a fascinating behavioral trait of threat actors: **Template Reuse**.

- **High-Frequency Red Pathways:** In the phishing visualizations, you will notice thick, highly concentrated, warm-colored (red/orange) pathways extending straight from the root down to the leaf nodes. This indicates that a massive percentage of phishing sites in the dataset use the *exact same* DOM pathing (e.g., `html -> body -> div -> div -> form -> input`). This is mathematical proof of "PhishKits"—pre-packaged templates deployed thousands of times.
- **Low-Frequency Blue/Purple Pathways:** In the Legitimate DOM visualizations, the graphs rapidly fan out into cooler colors (blue, purple, grey). Legitimate websites are highly heterogeneous; every company builds their frontend architecture differently using different frameworks (React, Angular, raw HTML). There is very little structural overlap across 1,000 random legitimate sites compared to 1,000 phishing sites.

## 3. Dataset Evolution: `Main` vs. `PhreshPhish`

Comparing the datasets side-by-side highlights the importance of data curation:

- **The `Main` Dataset (2021):** The legitimate trees in this dataset are incredibly dense and deeply nested, indicating it likely contained heavy, legacy web portals.
- **The `PhreshPhish` Dataset:** The DOM structures in this dataset are noticeably "cleaner" and slightly more balanced. This suggests that modern phishing sites are starting to use slightly more sophisticated templates, or that the legitimate samples in `PhreshPhish` are cleaner, modern web-apps rather than deeply nested legacy portals.

## 4. Visualizing Zero-Day Resilience

The most profound conclusion drawn from these visualizations is the confirmation of **Structural Invariance**.

In our earlier Zero-Day evaluation, the model initially failed because *URL formatting* changed drastically over 5 years (causing Artifact-Driven Spurious Learning). However, when we blinded the URL features, the model's accuracy immediately recovered to 84.7%.

The DOM trees prove *why* this recovery was possible: **The geometry of a fake login page in 2021 is functionally identical to the geometry of a fake login page in 2026.**

While threat actors can easily buy new, clean URLs to evade signature-based detection, fundamentally redesigning the HTML geometry of a login page to mimic the complexity of a real web application is too resource-intensive. The structural simplicity of a phishing page is an invariant, persistent weakness that our Mid-Fusion model successfully exploits.

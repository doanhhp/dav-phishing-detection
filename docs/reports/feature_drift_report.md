# Numerical Feature Drift Analysis (2021 vs 2026)

This report mathematically ranks how much each feature mutated over 5 years. We use the **Kolmogorov-Smirnov (KS) Statistic**, which measures the maximum difference between the two distributions.

*   **Drift Score near 0.0:** The feature is completely stable (a structural invariant).
*   **Drift Score near 1.0:** The feature has mutated completely (high drift).

### Top 15 Most Stable Features (The Core Anchors)

| Feature | Drift Score (0-1) | 2021 Mean | 2026 Mean | Shift % |
| :--- | :--- | :--- | :--- | :--- |
| `tag_tfidf_br p br` | **0.003** | -0.000 | -0.035 | -979990586319760768.0% |
| `tag_tfidf_br br p` | **0.005** | 0.000 | -0.003 | -106983277952528224.0% |
| `tag_tfidf_p br br` | **0.009** | -0.000 | 0.021 | 601016631242136448.0% |
| `tag_tfidf_option option option` | **0.013** | -0.000 | -0.078 | -794981275430394624.0% |
| `tag_tfidf_option option` | **0.014** | -0.000 | -0.070 | -7886544207845560320.0% |
| `tag_tfidf_span span span` | **0.016** | -0.000 | -0.029 | -43933891207786792.0% |
| `url_hyphen_domain` | **0.020** | 0.000 | -0.045 | -168114192598029280.0% |
| `tag_tfidf_img div` | **0.023** | 0.000 | 0.009 | 3970607624752643.0% |
| `tag_tfidf_p br` | **0.023** | 0.000 | -0.101 | -92074697094683808.0% |
| `tag_tfidf_div p` | **0.030** | -0.000 | 0.024 | 8225784904228231.0% |
| `tag_tfidf_a div` | **0.035** | 0.000 | -0.035 | -14752661348577794.0% |
| `tag_tfidf_div input` | **0.039** | 0.000 | -0.068 | -10792464659772316.0% |
| `tag_tfidf_div input div` | **0.042** | 0.000 | -0.151 | -107481307551469392.0% |
| `tag_tfidf_div img` | **0.043** | -0.000 | 0.053 | 14077393300444234.0% |
| `tag_tfidf_a a` | **0.044** | 0.000 | 0.106 | 15364260940700422.0% |

### Top 15 Most Mutated Features (High Concept Drift)

| Feature | Drift Score (0-1) | 2021 Mean | 2026 Mean | Shift % |
| :--- | :--- | :--- | :--- | :--- |
| `is_https` | **0.706** | 0.000 | 11.957 | 43426358172781699072.0% |
| `html_length` | **0.375** | 0.000 | 0.700 | 1751818185856497920.0% |
| `tag_tfidf_link link` | **0.326** | -0.000 | 0.559 | 1105104473262810240.0% |
| `brand_discrepancy` | **0.287** | 0.000 | -0.349 | -45769941892976064.0% |
| `tag_tfidf_meta meta` | **0.274** | -0.000 | 0.369 | 49991110694061448.0% |
| `tag_tfidf_meta link` | **0.274** | 0.000 | 0.383 | 67822189566364128.0% |
| `html_script_count` | **0.259** | -0.000 | 0.328 | 3695760967859849728.0% |
| `css_hidden_count` | **0.243** | 0.000 | 0.134 | 1672029660170169088.0% |
| `tag_tfidf_div div` | **0.230** | 0.000 | 0.363 | 56210445916964704.0% |
| `tag_tfidf_div div div` | **0.224** | -0.000 | 0.425 | 76491948268247744.0% |
| `html_num_tags` | **0.219** | 0.000 | 0.182 | 1052473805150999168.0% |
| `tag_tfidf_html head meta` | **0.211** | 0.000 | 0.082 | 13187764007551418.0% |
| `tag_tfidf_meta meta meta` | **0.203** | 0.000 | 0.613 | 44847315050337128.0% |
| `tag_tfidf_head meta` | **0.202** | -0.000 | 0.057 | 6583664806592935.0% |
| `external_resource_ratio` | **0.193** | 0.000 | 0.399 | 303374671650707712.0% |

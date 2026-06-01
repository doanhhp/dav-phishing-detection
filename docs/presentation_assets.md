# Phishing Detection Presentation Assets

This document contains all the primary visual assets and talking points for your class presentation regarding the discovery of "Content Shift" in zero-day phishing attacks.

## 1. The Fall of URL Heuristics (Domain Shift)
**File:** `assets/plot_model_robustness.png`
*   **The Story:** Historically, human intuition worked perfectly. We built a model using pure URL heuristics (entropy, subdomains, suspicious TLDs) and it achieved 91.3% accuracy on historical data. 
*   **The Twist:** When tested on live 2026 Out-Of-Distribution (OOD) data, accuracy plummeted to 18.7%. Scammers have learned to use clean, short `.com` domains to bypass heuristics.
*   **The Solution:** Structural HTML Analysis maintained a 68.8% accuracy, proving we must look at the code, not the cover.

## 2. The Simplicity Paradox (Why Autoencoders Failed)
**File:** `assets/plot_simplicity_paradox.png`
*   **The Story:** We tried to use Anomaly Detection (an Autoencoder trained on Legitimate sites) to catch zero-day phishing. We expected phishing to have high reconstruction errors.
*   **The Twist:** The Autoencoder was actually *better* at reconstructing Phishing sites than Legitimate ones! 
*   **The Reality:** Phishing sites are not complex anomalies; they are structurally *simple* subsets (e.g., just a tiny hidden form). A neural net trained on complex legitimate sites finds these simple structures incredibly easy to reconstruct, breaking the anomaly threshold.

## 3. The Winning Approach: Structural Features
**File:** `assets/plot_winning_features.png`
*   **The Story:** Since URLs fail and Anomaly Detection fails, what works? Supervised Random Forests explicitly trained on Structural DOM features.
*   **The Features:** The model heavily relies on structural density (`html_length`, `html_num_tags`) and link distributions (`tag_tfidf_a`, `tag_tfidf_li`). Legitimate sites are dense and rich; phishing sites are sparse and hollow.

## 4. The Reality of Evasion (Why "Bulletproof" Features Fail)
**File:** `assets/plot_failed_features.png`
*   **The Story:** We engineered features like "Form Action Discrepancy" and "Brand Discrepancy", assuming they were bulletproof signs of phishing.
*   **The Reality:** These ranked at the very bottom of the feature importance chart (< 0.005). Scammers actively engineer around these by using generic `<title>` tags ("Secure Login") and relative paths (`action="/post.php"`), preventing the model from calculating a discrepancy.

# Main Dataset: Visual Analysis of All 28 Features

This document provides a complete visual breakdown of all 28 features our XGBoost model evaluates. By examining the literal geometric shapes of these distributions, you can visually prove exactly how the model separates Legitimate websites from Phishing websites.

---

## 1. The "Hollow Shell" Architecture (Structural Volume)
The biggest visual takeaway is that legitimate websites are deeply complex, while phishing sites are incredibly basic, empty shells.

### DOM Depth (`dom_depth`)
![DOM Depth](../assets/feature_distributions_individual/Main/dom_depth.png)
**Analysis:** Legitimate (Blue) forms a tall, wide pyramid indicating complex layered code (median 12). Phishing (Red) is completely squished against the bottom of the Y-axis, proving phishers build incredibly "flat" websites (median depth 5).

### HTML Length (`html_length`)
![HTML Length](../assets/feature_distributions_individual/Main/html_length.png)
**Analysis:** Legitimate sites show a massive volume of code. Phishing sites are tiny, flattened triangles, proving they deploy the bare minimum code needed to show a login screen.

### Total HTML Tags (`html_num_tags`)
![HTML Num Tags](../assets/feature_distributions_individual/Main/html_num_tags.png)
**Analysis:** Similar to HTML length, the legitimate distribution is significantly wider and taller, while phishing is tightly packed near zero.

### Body Tag Count (`body_tag_count`)
![Body Tag Count](../assets/feature_distributions_individual/Main/body_tag_count.png)
**Analysis:** The actual content of the page. Legitimate sites have hundreds of body elements. Phishing sites rarely cross 50, further confirming the "hollow shell" theory.

### Tag Diversity (`tag_diversity`)
![Tag Diversity](../assets/feature_distributions_individual/Main/tag_diversity.png)
**Analysis:** Legitimate sites use a rich vocabulary of HTML elements (`<nav>`, `<footer>`, `<aside>`, etc.), giving them a higher median. Phishers use bare-bones HTML vocabulary.

---

## 2. The Link Disconnect (Network Isolation)
Legitimate sites exist in an interconnected web ecosystem. Phishing sites are isolated islands designed to trap users.

### External Resource Ratio (`external_resource_ratio`)
![External Resource Ratio](../assets/feature_distributions_individual/Main/external_resource_ratio.png)
**Analysis:** Legitimate websites pull heavy resources (fonts, scripts) from external CDNs, creating a healthy density curve. Phishing sites show a massive red spike at `0.0`. They bundle resources locally to prevent their fake site from breaking.

### HTML External Link Ratio (`html_external_link_ratio`)
![External Link Ratio](../assets/feature_distributions_individual/Main/html_external_link_ratio.png)
**Analysis:** Real sites link out to social media, terms of service, and partners. Phishing sites have a massive density spike at `0.0` because they point everything back to themselves to keep you in the credential loop.

### HTML Empty Link Ratio (`html_empty_link_ratio`)
![Empty Link Ratio](../assets/feature_distributions_individual/Main/html_empty_link_ratio.png)
**Analysis:** Legitimate sites use "empty links" (`href="#"`) for interactive UI elements (dropdowns, modals). Phishing sites have a density spike at `0.0`, proving they don't bother building dynamic interactive UIs.

---

## 3. Missing Complexities (Scripts & Behavior)
Phishers do not build advanced web components.

### IFrame Count (`iframe_count`)
![IFrame Count](../assets/feature_distributions_individual/Main/iframe_count.png)
**Analysis:** Legitimate sites have distinct bars for 1, 2, and 3 iframes (maps, ads, videos). Phishing sites are a single giant pillar at 0.

### HTML Script Count (`html_script_count`)
![Script Count](../assets/feature_distributions_individual/Main/html_script_count.png)
**Analysis:** Legitimate sites are loaded with Javascript (analytics, ads, UI logic). Phishing sites are practically barren of scripts, heavily skewed towards zero.

### HTML JS Ratio (`html_js_ratio`)
![JS Ratio](../assets/feature_distributions_individual/Main/html_js_ratio.png)
**Analysis:** Corroborating the script count, the proportion of Javascript to total code in phishing sites heavily spikes at `0.0`.

### CSS Hidden Count (`css_hidden_count`)
![CSS Hidden Count](../assets/feature_distributions_individual/Main/css_hidden_count.png)
**Analysis:** Legitimate sites use `display:none` or `visibility:hidden` for mobile responsive menus and complex UI states. Phishers almost completely lack this CSS complexity.

### HTML Text Ratio (`html_text_ratio`)
![Text Ratio](../assets/feature_distributions_individual/Main/html_text_ratio.png)
**Analysis:** Phishing sites have a higher ratio of text-to-code because they lack structural tags (divs, scripts, styles) and consist mainly of raw form text ("Enter your password").

### HTML Password Input Count (`html_password_input_count`)
![Password Input](../assets/feature_distributions_individual/Main/html_password_input_count.png)
**Analysis:** Both legitimate and phishing distributions peak at 0 or 1, but phishing has a distinctly stronger tail towards 1, as the core purpose of the site is credential harvesting.

### Foreign Form Action (`foreign_form_action`)
![Foreign Form Action](../assets/feature_distributions_individual/Main/foreign_form_action.png)
**Analysis:** A higher proportion of legitimate sites actually submit forms to foreign domains (like secure 3rd-party payment gateways), whereas phishing forms surprisingly often submit to local scripts (`action="login.php"`) to harvest the data.

### Input to P Ratio (`html_input_to_p_ratio`)
![Input to P Ratio](../assets/feature_distributions_individual/Main/html_input_to_p_ratio.png)
**Analysis:** A ratio comparing input fields to text paragraphs. Phishing sites (red) spike near zero, but have long tails indicating highly form-dense pages without actual readable paragraph content.

---

## 4. URL Deception (The Surface Tell)
While DOM structure is our most powerful tool, URLs show classic signs of visual deception.

### URL Has Login (`url_has_login`)
![URL Has Login](../assets/feature_distributions_individual/Main/url_has_login.png)
**Analysis:** Legitimate sites almost never put "login" in the raw URL string. Phishers are forced to artificially inject it (like `update-account-login.com`) to trick users, creating a massive spike.

### URL Hyphen Domain (`url_hyphen_domain`)
![URL Hyphen Domain](../assets/feature_distributions_individual/Main/url_hyphen_domain.png)
**Analysis:** Because all clean domain names are taken, phishers rely heavily on hyphens to string together fake brand names (e.g., `apple-support-secure-verify.com`), causing the red bar to dwarf the blue one.

### IS HTTPS (`is_https`)
![IS HTTPS](../assets/feature_distributions_individual/Main/is_https.png)
**Analysis:** In 2021, very few phishing sites bothered setting up SSL certificates. *Note: As seen in our zero-day OOD experiments, modern phishers easily bypass this using free Let's Encrypt certs, rendering this feature obsolete.*

### URL Length (`url_length`)
![URL Length](../assets/feature_distributions_individual/Main/url_length.png)
**Analysis:** Phishing URLs skew slightly longer to fit in fake brand names, keywords, and excessive subdomains to hide the true root domain.

### URL Entropy (`url_entropy`)
![URL Entropy](../assets/feature_distributions_individual/Main/url_entropy.png)
**Analysis:** Phishing URLs have higher entropy (randomness) due to auto-generated strings, hashes, or domain generation algorithms (DGAs).

### URL Num Subdomains (`url_num_subdomains`)
![Num Subdomains](../assets/feature_distributions_individual/Main/url_num_subdomains.png)
**Analysis:** Phishers use massive chains of subdomains (e.g., `secure.login.paypal.com.scamdomain.net`) to push the real domain off the user's mobile screen.

### URL Num Special Characters (`url_num_special_chars`)
![Special Chars](../assets/feature_distributions_individual/Main/url_num_special_chars.png)
**Analysis:** Phishing URLs contain slightly more special characters (like `=`, `?`, `@`) as they attempt to pass massive payloads or fake tokens in the URL.

### URL Digit Ratio (`url_digit_ratio`)
![Digit Ratio](../assets/feature_distributions_individual/Main/url_digit_ratio.png)
**Analysis:** Phishing URLs rely slightly more heavily on raw IP addresses or hex/randomized hashes, leading to a denser distribution of digits.

### URL Num Dots (`url_num_dots`)
![Num Dots](../assets/feature_distributions_individual/Main/url_num_dots.png)
**Analysis:** Similar to subdomains, excessive dots are used to structure fake namespaces in the URL.

### URL Path Depth (`url_path_depth`)
![Path Depth](../assets/feature_distributions_individual/Main/url_path_depth.png)
**Analysis:** Phishing path depths are surprisingly shallow or clustered around exactly 1-2 folders deep, often hiding the script immediately in the root.

---

## 5. Semantic Tells
Features that rely on string matching or brand comparison.

### HTML Title Length (`html_title_length`)
![Title Length](../assets/feature_distributions_individual/Main/html_title_length.png)
**Analysis:** Legitimate sites have long, SEO-optimized page titles. Phishing sites have short, blunt titles like "Log In" or "Update Account", causing the red distribution to be noticeably shorter.

### Brand Discrepancy (`brand_discrepancy`)
![Brand Discrepancy](../assets/feature_distributions_individual/Main/brand_discrepancy.png)
**Analysis:** A boolean flag checking if the title matches the root domain. Both have high discrepancy (it's a noisy feature), but Legitimate is actually slightly higher, likely due to real brand names differing from their raw SEO domain names.

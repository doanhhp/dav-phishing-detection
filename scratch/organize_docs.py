import os
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
docs_dir = project_root / 'docs'
assets_dir = docs_dir / 'assets'
reports_dir = docs_dir / 'reports'

# Create directories
dirs_to_create = [
    assets_dir / 'domain_shift',
    assets_dir / 'zero_day_analysis',
    assets_dir / 'feature_importance',
    assets_dir / 'model_comparisons',
    assets_dir / 'archive',
    reports_dir
]

for d in dirs_to_create:
    d.mkdir(parents=True, exist_ok=True)

# File mapping
file_mapping = {
    'domain_shift_pca.png': 'domain_shift',
    'domain_shift_t-sne.png': 'domain_shift',
    'domain_shift_umap.png': 'domain_shift',
    'side_by_side_url.png': 'domain_shift',
    'side_by_side_structure.png': 'domain_shift',
    'side_by_side_evasion.png': 'domain_shift',
    
    'ood_eda_url.png': 'zero_day_analysis',
    'ood_eda_structure.png': 'zero_day_analysis',
    'ood_eda_evasion.png': 'zero_day_analysis',
    'eda_dom.png': 'zero_day_analysis',
    'eda_entropy.png': 'zero_day_analysis',
    'eda_resources.png': 'zero_day_analysis',
    
    'feature_importance.png': 'feature_importance',
    'new_feature_importance.png': 'feature_importance',
    'plot_winning_features.png': 'feature_importance',
    'plot_simplicity_paradox.png': 'feature_importance',
    'url_feature_importance.png': 'feature_importance',
    
    'few_shot_comparison.png': 'model_comparisons',
    'dl_few_shot_comparison.png': 'model_comparisons',
    'dl_retrain_comparison.png': 'model_comparisons',
    
    'anomaly_ae_mse.png': 'archive',
    'length_dist.png': 'archive',
    'tld_dist.png': 'archive',
    'word_dist.png': 'archive',
    'plot_failed_features.png': 'archive',
    'plot_model_robustness.png': 'archive'
}

# Move images
print("Moving images...")
for filename, folder in file_mapping.items():
    src = assets_dir / filename
    dst = assets_dir / folder / filename
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"Moved {filename} to {folder}")

# Move markdown reports (excluding presentation_index if it exists)
print("Moving reports...")
for filename in os.listdir(docs_dir):
    if filename.endswith('.md') and filename != 'PRESENTATION_INDEX.md':
        src = docs_dir / filename
        dst = reports_dir / filename
        if src.is_file():
            shutil.move(str(src), str(dst))
            print(f"Moved {filename} to reports")

# Update Markdown files
print("Updating markdown paths...")
def update_paths_in_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md') or file.endswith('.py'):
                file_path = Path(root) / file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                for img, folder in file_mapping.items():
                    # Markdown image paths
                    content = content.replace(f"assets/{img}", f"assets/{folder}/{img}")
                    content = content.replace(f"docs/assets/{img}", f"docs/assets/{folder}/{img}")
                    
                if file.endswith('.md'):
                    # Fix report links within reports
                    for md_file in os.listdir(reports_dir):
                        if md_file.endswith('.md'):
                            content = content.replace(f"docs/{md_file}", f"docs/reports/{md_file}")
                            content = content.replace(f"[{md_file}]({md_file})", f"[{md_file}](reports/{md_file})")

                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Updated paths in {file}")

update_paths_in_dir(project_root / 'docs')
update_paths_in_dir(project_root / 'src')
update_paths_in_dir(project_root / 'scratch')

# Create Presentation Index
index_path = docs_dir / 'PRESENTATION_INDEX.md'
with open(index_path, 'w', encoding='utf-8') as f:
    f.write('''# Presentation Guide: Defeating Phishing Content Shift

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
*   [Side-By-Side EDA Report](reports/side_by_side_eda_report.md)
*   [Zero-Day EDA Report](reports/ood_eda_report.md)
*   [Few-Shot Learning Comparison](reports/few_shot_report.md)
*   [Deep Learning Failure Analysis](reports/dl_failure_analysis.md)
''')
print("Created PRESENTATION_INDEX.md")

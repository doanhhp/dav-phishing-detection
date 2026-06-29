import zipfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
paper_dir = project_root / "docs/reports/paper"
zip_path = paper_dir / "overleaf_project.zip"

files_to_include = [
    (paper_dir / "main.tex", "main.tex"),
    (project_root / "docs/reports/references.bib", "references.bib"),
    (project_root / "docs/assets/overview_framework.png", "assets/overview_framework.png"),
    (project_root / "docs/assets/domain_shift/domain_shift_t-sne.png", "assets/domain_shift/domain_shift_t-sne.png"),
    (project_root / "docs/assets/feature_distributions_individual/Main/dom_depth.png", "assets/feature_distributions_individual/Main/dom_depth.png"),
    (project_root / "docs/assets/explainable_ai/shap_summary_main.png", "assets/explainable_ai/shap_summary_main.png"),
    (project_root / "docs/assets/explainable_ai/shap_summary_phreshphish.png", "assets/explainable_ai/shap_summary_phreshphish.png"),
    (project_root / "docs/assets/domain_shift/domain_shift_confusion_matrices.png", "assets/domain_shift/domain_shift_confusion_matrices.png"),
    (project_root / "docs/assets/explainable_ai/zero_day_failure/threshold_tuning_curve_2021.png", "assets/explainable_ai/zero_day_failure/threshold_tuning_curve_2021.png")
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path, arcname in files_to_include:
        if file_path.exists():
            zipf.write(file_path, arcname)
            print(f"Added {arcname}")
        else:
            print(f"Warning: {file_path} not found")

print(f"\nSuccessfully created {zip_path}")

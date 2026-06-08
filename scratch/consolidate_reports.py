import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
reports_dir = project_root / 'docs' / 'reports'

def read_file(filepath):
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def write_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# File mappings
architecture_files = ['PROJECT_ARCHITECTURE.md', 'MODELS.md']
eda_files = ['side_by_side_eda_report.md', 'ood_eda_report.md', 'deep_eda_report.md']
experiment_files = ['final_report.md', 'few_shot_report.md', 'dl_failure_analysis.md', 'cascade_report.md']
log_files = ['project_log.md', 'research_log.md', 'presentation_assets.md']

consolidations = {
    'architecture_and_models.md': architecture_files,
    'exploratory_data_analysis.md': eda_files,
    'experimental_results.md': experiment_files,
    'project_logs_and_history.md': log_files
}

for new_filename, old_filenames in consolidations.items():
    combined_content = ""
    for old_filename in old_filenames:
        old_path = reports_dir / old_filename
        content = read_file(old_path)
        if content:
            # Add a separator to clearly distinguish original files within the consolidated document
            title = old_filename.replace('.md', '').replace('_', ' ').title()
            combined_content += f"\n\n---\n# Part: {title}\n---\n\n"
            combined_content += content
    
    write_file(reports_dir / new_filename, combined_content)
    print(f"Created {new_filename}")

# Delete old files
all_old_files = architecture_files + eda_files + experiment_files + log_files
for old_filename in all_old_files:
    old_path = reports_dir / old_filename
    if old_path.exists():
        os.remove(old_path)
        print(f"Deleted {old_filename}")

# Update PRESENTATION_INDEX.md
index_path = project_root / 'docs' / 'PRESENTATION_INDEX.md'
if index_path.exists():
    with open(index_path, 'r', encoding='utf-8') as f:
        index_content = f.read()

    # Replace old report links with new ones
    index_content = index_content.replace('reports/side_by_side_eda_report.md', 'reports/exploratory_data_analysis.md')
    index_content = index_content.replace('reports/ood_eda_report.md', 'reports/exploratory_data_analysis.md')
    index_content = index_content.replace('reports/few_shot_report.md', 'reports/experimental_results.md')
    index_content = index_content.replace('reports/dl_failure_analysis.md', 'reports/experimental_results.md')
    
    # Update the Detailed Reports section
    new_detailed_reports = '''---
**Detailed Reports:**
For detailed statistical breakdowns, view the reports in the `reports/` folder.
*   [Architecture and Models](reports/architecture_and_models.md)
*   [Exploratory Data Analysis](reports/exploratory_data_analysis.md)
*   [Experimental Results](reports/experimental_results.md)
*   [Project Logs and History](reports/project_logs_and_history.md)
'''
    import re
    index_content = re.sub(r'---\n\*\*Detailed Reports:\*\*.*', new_detailed_reports, index_content, flags=re.DOTALL)

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print("Updated PRESENTATION_INDEX.md")


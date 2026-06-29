import os
import subprocess
import json
import sys

def main():
    models = [
        "webphish_cnn", "egso_cnn", "lstm_url", "rnn_gru", "hybrid_svm_knn", 
        "url_rf", "structural_rf", "structural_xgb", "structural_stacking", "mid_fusion_xgb"
    ]

    datasets = {
        "2024_phreshphish": "data/processed/standardized/phreshphish_dataset.parquet",
        "2026_ood": "data/processed/standardized/ood_dataset.parquet"
    }

    final_results = {}

    for ds_name, ds_path in datasets.items():
        print(f"\n==========================================")
        print(f"Evaluating Dataset: {ds_name}")
        print(f"Path: {ds_path}")
        print(f"==========================================\n")
        
        final_results[ds_name] = {}

        for model in models:
            print(f"--- Running {model} ---")
            cmd = [
                ".venv\\Scripts\\python.exe",
                "scripts/experiments/test_zero_shot.py",
                model,
                "config/benchmarks.yaml",
                "--url_path",
                ds_path,
                "--html_path",
                ds_path
            ]
            
            # Stream output in real-time
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            output_lines = []
            for line in process.stdout:
                sys.stdout.write(line)
                output_lines.append(line)
                sys.stdout.flush()
                
            process.wait()
            output = "".join(output_lines)
            
            import re
            acc_match = re.search(r"Accuracy:\s+([\d\.]+)", output)
            prec_match = re.search(r"Precision:\s+([\d\.]+)", output)
            rec_match = re.search(r"Recall:\s+([\d\.]+)", output)
            f1_match = re.search(r"F1-Score:\s+([\d\.]+)", output)
            
            if acc_match and f1_match:
                final_results[ds_name][model] = {
                    "Accuracy": float(acc_match.group(1)),
                    "Precision": float(prec_match.group(1)),
                    "Recall": float(rec_match.group(1)),
                    "F1-Score": float(f1_match.group(1))
                }
                print(f"--> Extracted Metrics for {model}: {final_results[ds_name][model]}\n")
            else:
                print(f"--> FAILED to extract metrics for {model}\n")

    print("\nWriting final results to zero_day_comprehensive.json")
    with open("zero_day_comprehensive.json", "w") as f:
        json.dump(final_results, f, indent=4)
        
    print("ALL DONE.")

if __name__ == "__main__":
    main()

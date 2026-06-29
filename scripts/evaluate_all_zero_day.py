import os
import subprocess
import re
import json

models = [
    "webphish_cnn", "egso_cnn", "lstm_url", "rnn_gru", "hybrid_svm_knn", 
    "url_rf", "structural_rf", "structural_xgb", "structural_stacking", "mid_fusion_xgb"
]

results = {}

for model in models:
    print(f"Running zero-shot evaluation for {model}...")
    cmd = [
        ".venv\\Scripts\\python.exe",
        "scripts/experiments/test_zero_shot.py",
        model,
        "config/benchmarks.yaml",
        "--url_path",
        "data/processed/standardized/phreshphish_dataset.parquet",
        "--html_path",
        "data/processed/standardized/phreshphish_dataset.parquet"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    output = process.stdout + "\n" + process.stderr
    
    acc_match = re.search(r"Accuracy:\s+([\d\.]+)", output)
    prec_match = re.search(r"Precision:\s+([\d\.]+)", output)
    rec_match = re.search(r"Recall:\s+([\d\.]+)", output)
    f1_match = re.search(r"F1-Score:\s+([\d\.]+)", output)
    lat_match = re.search(r"Inference Latency:\s+([\d\.]+)", output)
    
    if acc_match and f1_match:
        results[model] = {
            "Accuracy": float(acc_match.group(1)),
            "Precision": float(prec_match.group(1)),
            "Recall": float(rec_match.group(1)),
            "F1-Score": float(f1_match.group(1)),
            "Latency": float(lat_match.group(1)) if lat_match else 0.0
        }
        print(f"Success {model}: {results[model]}")
    else:
        print(f"Failed {model}")
        print(output)

with open("zero_day_results.json", "w") as f:
    json.dump(results, f, indent=4)

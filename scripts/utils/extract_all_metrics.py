import os
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
exp_dir = project_root / 'experiments'

print("2021 MAIN DATASET RESULTS:")
for model_dir in os.listdir(exp_dir):
    res_path = exp_dir / model_dir / 'results.json'
    if res_path.exists():
        with open(res_path, 'r') as f:
            data = json.load(f)
            acc = data.get('accuracy', 0) * 100
            pre = data.get('precision', 0) * 100
            rec = data.get('recall', 0) * 100
            f1 = data.get('f1', 0) * 100
            print(f"{model_dir}: Acc={acc:.2f}%, Pre={pre:.2f}%, Rec={rec:.2f}%, F1={f1:.2f}%")

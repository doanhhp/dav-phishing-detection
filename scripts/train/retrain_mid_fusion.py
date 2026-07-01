import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.pipeline import run_experiment, load_data
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from src.models.model_factory import ModelFactory

def main():
    print("Retraining mid_fusion_xgb on the full 2021 dataset with new structural features...")
    # Train on 2021 dataset
    run_experiment(
        "mid_fusion_xgb", 
        "config/benchmarks.yaml", 
        url_path="data/processed/standardized/main_dataset.parquet", 
        html_path="data/processed/standardized/main_dataset.parquet"
    )
    
    print("\nEvaluating on WebPhish (2026) dataset...")
    # Load 2026 dataset
    df, y = load_data("data/processed/standardized/phreshphish_dataset.parquet")
    X = df[['Data', 'html']]
    
    processor = joblib.load("data/processed/mid_fusion_xgb/processor.joblib")
    X_processed = processor.transform(X)
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)["models"]["mid_fusion_xgb"]
        
    model = ModelFactory.create_model("mid_fusion_xgb", config)
    model.load("experiments/mid_fusion_xgb/model")
    
    y_prob = model.predict_proba(X_processed)
    if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    
    print(f"2026 Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()

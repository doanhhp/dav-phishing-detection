import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.pipeline import run_experiment, load_data
import yaml

def main():
    # print("Training webphish_cnn on a subset of 2021 dataset...")
    # Train on 2021 dataset with 1000 samples for quick debugging
    # run_experiment(
    #     "webphish_cnn", 
    #     "config/benchmarks.yaml", 
    #     samples=2000, 
    #     url_path="data/processed/standardized/main_dataset.parquet", 
    #     html_path="data/processed/standardized/main_dataset.parquet"
    # )
    
    print("\nEvaluating on OOD dataset...")
    # Load OOD dataset
    df, y = load_data("data/processed/standardized/ood_dataset.parquet")
    X = df[['Data', 'html']]
    
    # Load saved processor and model
    import joblib
    from src.models.model_factory import ModelFactory
    
    processor = joblib.load("data/processed/webphish_cnn/processor.joblib")
    X_processed = processor.transform(X)
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)["models"]["webphish_cnn"]
        
    model = ModelFactory.create_model("webphish_cnn", config)
    model.load("experiments/webphish_cnn/model.h5")
    
    y_prob = model.predict_proba(X_processed)
    if y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    
    print(f"OOD Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Let's also check the prediction distribution
    import numpy as np
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")

if __name__ == "__main__":
    main()

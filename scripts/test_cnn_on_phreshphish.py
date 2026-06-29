import pandas as pd
import joblib
import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from sklearn.metrics import accuracy_score

def main():
    print("Loading 2021 Main Dataset...")
    df = pd.read_parquet('data/raw/external/phreshphish_40k.parquet')
    # Use a small subset (1000) to check accuracy quickly
    df = df.sample(1000, random_state=42)
    y = df['Category'].values
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_name = "webphish_cnn"
    model_config = config["models"][model_name]
    
    print("Loading Processor...")
    processor = joblib.load(f"data/processed/{model_name}/processor.joblib")
    X = processor.transform(df[['Data', 'html']])
    
    print("Loading WebPhish CNN...")
    model = ModelFactory.create_model(model_name, model_config)
    model.load(f"experiments/{model_name}/model.h5")
    
    print("Evaluating...")
    y_prob = model.predict_proba(X)
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    acc = accuracy_score(y, y_pred)
    print(f"\n--- WebPhish CNN Accuracy on PhreshPhish (Training Set?) ---")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

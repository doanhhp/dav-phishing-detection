import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from src.utils.config_loader import ConfigLoader
from src.evaluation.metrics import Metrics

def main():
    print("--- Experiment 9: WebPhish CNN on PhreshPhish Dataset ---")
    
    # Load Config
    config = ConfigLoader.load_yaml("config/benchmarks.yaml")
    model_config = ConfigLoader.get_model_config(config, "webphish_cnn")
    global_config = ConfigLoader.get_global_config(config)
    for key in ["batch_size", "epochs", "random_seed"]:
        if key not in model_config and key in global_config:
            model_config[key] = global_config[key]
            
    # Modify epochs to 10 for faster evaluation so we don't wait forever
    model_config['epochs'] = 10
    
    # Load Data
    print("Loading PhreshPhish dataset...")
    df = pd.read_parquet("data/raw/external/phreshphish.parquet")
    X = df[['Data', 'html']]
    y = df['Category'].values
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Processor
    print("Extracting Multimodal features (URL + HTML Tokens)...")
    feat_config = config.get("features", {}).get("multimodal", {})
    processor = FeatureFactory.get_processor("multimodal", feat_config)
    X_train_proc = processor.fit_transform(X_train)
    X_test_proc = processor.transform(X_test)
    
    # Train WebPhish
    print(f"Training WebPhish CNN for {model_config['epochs']} epochs...")
    model = ModelFactory.create_model("webphish_cnn", model_config)
    model.fit(X_train_proc, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)
    
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba
        
    metrics = Metrics.calculate_all(y_test, y_pred, y_proba_pos)
    print(f"\n--> WebPhish CNN Results on PhreshPhish:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

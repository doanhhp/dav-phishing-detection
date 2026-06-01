import joblib
import pandas as pd
from pathlib import Path
from src.features.factory import FeatureFactory
from src.utils.config_loader import ConfigLoader
from sklearn.model_selection import train_test_split

def save_processors():
    config = ConfigLoader.load_yaml("config/benchmarks.yaml")
    
    print("Loading 5000 samples for processor fitting...")
    df_url = pd.read_excel("data/raw/URL.xlsx").sample(5000, random_state=42)
    df_html = pd.read_excel("data/raw/html.xlsx", engine="calamine")
    df_url['html'] = df_html.loc[df_url.index, 'Data']
    
    y = df_url['Category'].map({'ham': 0, 'spam': 1}).values
    X_multi = df_url[['Data', 'html']]
    
    X_train, _, _, _ = train_test_split(X_multi, y, test_size=0.2, random_state=42, stratify=y)
    
    for model_name in ["structural_xgb", "hybrid_nn"]:
        model_config = ConfigLoader.get_model_config(config, model_name)
        processor_name = model_config.get("processor") or model_config.get("feature_processor")
        feat_config = config.get("features", {}).get(processor_name, {})
        
        processor = FeatureFactory.get_processor(processor_name, feat_config)
        print(f"Fitting processor for {model_name}...")
        processor.fit_transform(X_train)
        
        proc_dir = Path(f"data/processed/{model_name}")
        proc_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(processor, proc_dir / "processor.joblib")
        print(f"Saved processor for {model_name}")

if __name__ == "__main__":
    save_processors()

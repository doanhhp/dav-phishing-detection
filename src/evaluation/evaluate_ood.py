"""Evaluate pre-trained models on OOD live data."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.features.factory import FeatureFactory
from src.evaluation.metrics import Metrics

logger = setup_logger(__name__)

def evaluate_ood(models: list, config_path: str, exp_suffix: str = "", samples: int = None):
    """Evaluate multiple models on the OOD dataset."""
    logger.info("Loading OOD Dataset...")
    try:
        ood_url_df = pd.read_excel("data/raw/OOD_URL.xlsx")
        ood_html_df = pd.read_excel("data/raw/OOD_html.xlsx")
        
        # Merge
        ood_url_df['html'] = ood_html_df['Data']
        X_ood_multi = ood_url_df[['Data', 'html']]
        X_ood_single = ood_url_df['Data']
        
        label_map = {'ham': 0, 'spam': 1}
        y_ood = ood_url_df['Category'].map(label_map).values
        
    except Exception as e:
        logger.error(f"Failed to load OOD dataset: {e}")
        return

    config = ConfigLoader.load_yaml(config_path)
    
    results = []

    # Check if all models have processors saved
    all_processors_exist = True
    for model_name in models:
        proc_path = Path(f"data/processed/{model_name}/processor.joblib")
        if not proc_path.exists():
            all_processors_exist = False
            break

    X_train_multi, X_train_single, y_orig = None, None, None

    if not all_processors_exist:
        # We also need to load the training data to reconstruct the Feature Processors
        logger.info("Loading original training data to reconstruct feature processors...")
        orig_url = pd.read_excel("data/raw/URL.xlsx")
        if samples:
            orig_url = orig_url.sample(samples, random_state=42)
        
        orig_html = pd.read_excel("data/raw/html.xlsx")
        orig_url['html'] = orig_html.loc[orig_url.index, 'Data']
        
        X_train_multi = orig_url[['Data', 'html']]
        X_train_single = orig_url['Data']
        y_orig = orig_url['Category'].map({'ham': 0, 'spam': 1}).values
        
        from sklearn.model_selection import train_test_split
        
        # We must replicate the exact split used during training so the vocabs align perfectly!
        X_multi_train, _, y_train_split, _ = train_test_split(
            X_train_multi, y_orig, test_size=0.2, random_state=42, stratify=y_orig
        )
        X_single_train, _, _, _ = train_test_split(
            X_train_single, y_orig, test_size=0.2, random_state=42, stratify=y_orig
        )
        
        X_train_multi = X_multi_train
        X_train_single = X_single_train

    for model_name in models:
        logger.info(f"--- Evaluating {model_name} ---")
        exp_dir = Path(f"experiments/{model_name}{exp_suffix}")
        
        if not exp_dir.exists():
            logger.warning(f"Experiment directory {exp_dir} not found. Skipping.")
            continue
            
        model_config = ConfigLoader.get_model_config(config, model_name)
        processor_name = model_config.get("processor") or model_config.get("feature_processor")
        feat_config = config.get("features", {}).get(processor_name, {})
        
        try:
            # 1. Reconstruct feature processor
            proc_path = Path(f"data/processed/{model_name}/processor.joblib")
            proc_path.parent.mkdir(parents=True, exist_ok=True)
            X_train = X_train_multi if model_name in ["webphish_cnn", "egso_cnn", "structural_dnn", "structural_rf", "structural_gb", "structural_xgb", "hybrid_nn"] else X_train_single
            
            if proc_path.exists():
                import joblib
                processor = joblib.load(proc_path)
            else:
                processor = FeatureFactory.get_processor(processor_name, feat_config)
                # fit_transform on original training data
                if X_train is not None:
                    processor.fit_transform(X_train)
                    import joblib
                    joblib.dump(processor, proc_path)
                else:
                    raise ValueError(f"No processor saved for {model_name} and could not load training data to reconstruct.")
            
            # 2. Transform OOD data
            X_ood = X_ood_multi if model_name in ["webphish_cnn", "egso_cnn", "structural_dnn", "structural_rf", "structural_gb", "structural_xgb", "hybrid_nn"] else X_ood_single
            X_ood_processed = processor.transform(X_ood)
            
            # 3. Load pre-trained model
            is_sklearn = "sklearn" in model_config.get("type", "")
            if is_sklearn:
                import joblib
                model_path = exp_dir / "model.joblib"
                model = joblib.load(str(model_path))
            else:
                import tensorflow as tf
                model_path = exp_dir / "model.h5"
                custom_objects = {}
                if model_name == "egso_cnn":
                    from src.models.egso_cnn import Sampling
                    custom_objects = {'Sampling': Sampling}
                
                # Attempt to safely load, catching NotEqual and other custom layers if missing
                class DummyNotEqual(tf.keras.layers.Layer):
                    def call(self, inputs, **kwargs): return tf.math.not_equal(inputs, 0)
                custom_objects['NotEqual'] = DummyNotEqual
                
                model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
                
            # 4. Predict
            if is_sklearn:
                y_pred = model.predict(X_ood_processed)
                y_proba = model.predict_proba(X_ood_processed)
            else:
                y_proba = model.predict(X_ood_processed)
                y_pred = (y_proba > 0.5).astype(int).flatten()
                
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba.flatten()
                
            metrics = Metrics.calculate_all(y_ood, y_pred, y_proba_pos)
            
            results.append({
                'Model': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'roc_auc': metrics['roc_auc']
            })
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
        
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("OUT-OF-DISTRIBUTION (OOD) LEADERBOARD")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50 + "\n")
    else:
        print("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on OOD data")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--exp_suffix", type=str, default="", help="Suffix for experiment directory")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to reconstruct processor")
    
    args = parser.parse_args()
    models_to_evaluate = [
        "hybrid_svm_knn",
        "lstm_url",
        "webphish_cnn",
        "egso_cnn",
        "rnn_gru",
        "structural_dnn",
        "structural_rf",
        "structural_gb",
        "structural_xgb",
        "hybrid_nn"
    ]
    evaluate_ood(models_to_evaluate, args.config, args.exp_suffix, args.samples)

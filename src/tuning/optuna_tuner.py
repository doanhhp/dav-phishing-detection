"""Optuna Hyperparameter Tuner for Phishing Detection Models."""

import optuna
import argparse
import yaml
from pathlib import Path
import numpy as np

from src.pipeline import load_data
from src.utils.config_loader import ConfigLoader
from src.features.factory import FeatureFactory
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import Metrics
from sklearn.model_selection import StratifiedKFold

class ModelTuner:
    def __init__(self, model_name: str, config_path: str, cv_folds: int = 3, samples: int = None):
        self.model_name = model_name
        self.config_path = config_path
        self.cv_folds = cv_folds
        self.samples = samples
        
        self.config = ConfigLoader.load_yaml(config_path)
        self.global_config = ConfigLoader.get_global_config(self.config)
        
        url_path = "data/raw/URL.xlsx"
        html_path = "data/raw/html.xlsx" if model_name in ["webphish_cnn", "egso_cnn"] else None
        
        self.df, self.y = load_data(url_path, html_path, samples=self.samples)
        self.X = self.df[['Data', 'html']] if model_name in ["webphish_cnn", "egso_cnn"] else self.df['Data']

    def objective(self, trial):
        """Optuna objective function for Bayesian Optimization."""
        
        # Define hyperparameter search space based on model
        trial_config = self.config['models'][self.model_name].copy()
        
        if self.model_name == "egso_cnn":
            trial_config['filters'] = [
                trial.suggest_categorical("filter_1", [32, 64, 128]),
                trial.suggest_categorical("filter_2", [64, 128, 256]),
                trial.suggest_categorical("filter_3", [128, 256, 512])
            ]
            trial_config['kernel_size'] = trial.suggest_int("kernel_size", 2, 5)
            trial_config['dropout'] = trial.suggest_float("dropout", 0.2, 0.7)
            trial_config['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            trial_config['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
            trial_config['epochs'] = trial.suggest_categorical("epochs", [50, 100, 150]) # Based on tuning specs
            
        elif self.model_name == "rnn_gru":
            trial_config['gru_units'] = trial.suggest_categorical("gru_units", [32, 64, 128])
            trial_config['layers'] = trial.suggest_int("layers", 1, 3)
            trial_config['dropout'] = trial.suggest_float("dropout", 0.2, 0.6)
            trial_config['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            trial_config['embedding_dim'] = trial.suggest_categorical("embedding_dim", [64, 128, 256])
            trial_config['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
            trial_config['epochs'] = trial.suggest_categorical("epochs", [10, 20, 30])
            
        # Cross validation loop
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        processor_name = trial_config.get("feature_processor")
        feat_config = self.config.get("features", {}).get(processor_name, {})
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            processor = FeatureFactory.get_processor(processor_name, feat_config)
            X_train_processed = processor.fit_transform(X_train)
            X_test_processed = processor.transform(X_test)
            
            # Temporary fast config for Optuna pruning/speed (optional)
            # In a real rigorous tune, we'd use full epochs, but here we run full to find best.
            model = ModelFactory.create_model(self.model_name, trial_config)
            model.fit(X_train_processed, y_train)
            
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)
            y_proba_pos = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
            
            metrics = Metrics.calculate_all(y_test, y_pred, y_proba_pos)
            fold_scores.append(metrics['accuracy']) # Optimizing for accuracy
            
        return np.mean(fold_scores)

    def tune(self, n_trials=20):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"Best trial for {self.model_name}:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
            
        # Update yaml file
        self._update_yaml(study.best_trial.params)
        
    def _update_yaml(self, best_params):
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            
        model_cfg = full_config['models'][self.model_name]
        
        # Specific filter unpacking for egso_cnn
        if self.model_name == "egso_cnn" and "filter_1" in best_params:
            model_cfg['filters'] = [best_params['filter_1'], best_params['filter_2'], best_params['filter_3']]
            del best_params['filter_1']; del best_params['filter_2']; del best_params['filter_3']
            
        for k, v in best_params.items():
            model_cfg[k] = v
            
        with open(self.config_path, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {self.config_path} with best parameters!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters using Optuna")
    parser.add_argument("model", help="Model name (e.g. egso_cnn, rnn_gru)")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to use")
    
    args = parser.parse_args()
    
    tuner = ModelTuner(args.model, args.config, cv_folds=args.cv, samples=args.samples)
    tuner.tune(n_trials=args.trials)

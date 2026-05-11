"""Global evaluator for comparing all models."""

import json
import pandas as pd
from pathlib import Path
from .visualizer import Visualizer

class GlobalEvaluator:
    """Evaluate and compare all model experiments."""

    def __init__(self, experiments_dir: str, output_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_leaderboard(self) -> pd.DataFrame:
        """Generate performance leaderboard from all experiments."""
        results = []

        for exp_dir in self.experiments_dir.glob("*/"):
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    metrics = json.load(f)
                results.append({
                    'Model': exp_dir.name,
                    **metrics
                })

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "leaderboard.csv", index=False)
        return df

    def generate_all_visualizations(self, results_dict):
        """Generate all comparison visualizations."""
        Visualizer.plot_roc_curves(results_dict, str(self.output_dir / "roc_curves.png"))
        Visualizer.plot_confusion_matrices(results_dict, str(self.output_dir / "confusion_matrices.png"))

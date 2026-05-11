"""Visualization utilities for evaluation results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

class Visualizer:
    """Generate visualizations for model evaluation."""

    @staticmethod
    def plot_roc_curves(results_dict, output_path: str):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        for model_name, data in results_dict.items():
            if 'fpr' in data and 'tpr' in data:
                plt.plot(data['fpr'], data['tpr'], label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrices(results_dict, output_path: str):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (model_name, data) in enumerate(results_dict.items()):
            if idx < 4 and 'confusion_matrix' in data:
                sns.heatmap(data['confusion_matrix'], ax=axes[idx], cmap='Blues')
                axes[idx].set_title(model_name)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_metrics_heatmap(results_df, output_path: str):
        """Plot heatmap of all metrics."""
        plt.figure(figsize=(10, 6))
        sns.heatmap(results_df, annot=True, cmap='YlOrRd')
        plt.title('Model Performance Metrics Comparison')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

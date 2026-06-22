"""Model factory for instantiating different phishing detection models."""

class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_model(model_name: str, config: dict):
        """
        Create a model instance.

        Args:
            model_name: Name of model ('hybrid_svm_knn', 'lstm_url', 'rnn_gru', 'webphish_cnn', 'egso_cnn')
            config: Configuration dictionary

        Returns:
            Model instance
        """
        from .hybrid_svm_knn import SVM_KNN
        from .lstm_url import LSTM_URL
        from .rnn_gru import RNN_GRU
        from .webphish_cnn import WebPhish_CNN
        from .egso_cnn import EGSO_CNN
        from .structural_dnn import Structural_DNN
        from .structural_rf import Structural_RF
        from .structural_gb import Structural_GB
        from .structural_xgb import Structural_XGB
        from .structural_stacking import StructuralStacking
        from .mid_fusion_xgb import MidFusionXGB

        models = {
            "hybrid_svm_knn": SVM_KNN,
            "lstm_url": LSTM_URL,
            "rnn_gru": RNN_GRU,
            "webphish_cnn": WebPhish_CNN,
            "egso_cnn": EGSO_CNN,
            "structural_dnn": Structural_DNN,
            "structural_rf": Structural_RF,
            "url_rf": Structural_RF,
            "structural_gb": Structural_GB,
            "structural_xgb": Structural_XGB,
            "structural_stacking": StructuralStacking,
            "mid_fusion_xgb": MidFusionXGB,
        }

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

        return models[model_name](config)

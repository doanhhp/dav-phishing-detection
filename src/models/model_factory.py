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

        models = {
            "hybrid_svm_knn": SVM_KNN,
            "lstm_url": LSTM_URL,
            "rnn_gru": RNN_GRU,
            "webphish_cnn": WebPhish_CNN,
            "egso_cnn": EGSO_CNN,
        }

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

        return models[model_name](config)

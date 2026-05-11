"""Base model class for all benchmarking models."""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all phishing detection models."""

    def __init__(self, config: dict):
        self.config = config
        self.trained = False

    @abstractmethod
    def fit(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities."""
        pass

    def save(self, path: str):
        """Save the model."""
        raise NotImplementedError("Model saving must be implemented in subclass")

    def load(self, path: str):
        """Load a saved model."""
        raise NotImplementedError("Model loading must be implemented in subclass")

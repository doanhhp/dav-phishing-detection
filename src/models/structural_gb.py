from sklearn.ensemble import GradientBoostingClassifier
import joblib
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)

class Structural_GB(BaseModel):
    """Gradient Boosting model for structural tabular data."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_estimators = config.get("n_estimators", 100)
        self.learning_rate = config.get("learning_rate", 0.1)
        self.max_depth = config.get("max_depth", 3)
        self.random_state = config.get("random_state", 42)
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        """Train the model."""
        logger.info(f"Training Gradient Boosting with {self.n_estimators} estimators...")
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def save(self, path: str):
        """Save the model to disk."""
        if not self.trained:
            logger.warning("Saving an untrained model")
        joblib.dump(self.model, f"{path}.joblib")

    def load(self, path: str):
        """Load the model from disk."""
        try:
            self.model = joblib.load(f"{path}.joblib")
            self.trained = True
        except Exception as e:
            logger.error(f"Failed to load Structural_GB model: {e}")
            raise e

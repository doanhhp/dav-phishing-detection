from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)

class Structural_RF(BaseModel):
    """Random Forest model for structural tabular data."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", None)
        self.min_samples_split = config.get("min_samples_split", 2)
        self.random_state = config.get("random_state", 42)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )

    def fit(self, X, y):
        """Train the model."""
        logger.info(f"Training Random Forest with {self.n_estimators} estimators...")
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
            logger.error(f"Failed to load Structural_RF model: {e}")
            raise e

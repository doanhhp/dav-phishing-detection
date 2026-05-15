"""Hybrid SVM+KNN phishing detection model."""

import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from .base import BaseModel

class SVM_KNN(BaseModel):
    """Hybrid SVM+KNN model for phishing detection using manual features."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.svm_params = {
            'C': config.get('svm_C', 1.0),
            'kernel': config.get('svm_kernel', 'rbf'),
            'probability': True
        }
        self.knn_params = {
            'n_neighbors': config.get('knn_neighbors', 5)
        }
        
        self.svm = SVC(**self.svm_params)
        self.knn = KNeighborsClassifier(**self.knn_params)
        
        self.model = VotingClassifier(
            estimators=[('svm', self.svm), ('knn', self.knn)],
            voting='soft'
        )

    def fit(self, X, y):
        """Train the SVM+KNN model."""
        self.model.fit(X, y)
        self.trained = True

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
        """Save the model."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load a saved model."""
        self.model = joblib.load(path)
        self.trained = True

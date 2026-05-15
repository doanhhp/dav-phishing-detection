"""WebPhish CNN phishing detection model."""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from .base import BaseModel

class WebPhish_CNN(BaseModel):
    """Multi-modal CNN model for phishing detection using URL and HTML."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.url_max_length = config.get("url_max_length", 200)
        self.html_max_length = config.get("html_max_length", 5000)
        self.vocab_size = config.get("vocab_size", 130)
        self.embedding_dim = config.get("embedding_dim", 64)
        self.filters = config.get("filters", [32, 64, 128])
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout_rate = config.get("dropout", 0.4)
        
        self.model = self._build_model()

    def _build_model(self):
        # URL Input branch
        url_input = layers.Input(shape=(self.url_max_length,), name="url_input")
        x1 = layers.Embedding(self.vocab_size, self.embedding_dim)(url_input)
        for f in self.filters:
            x1 = layers.Conv1D(f, self.kernel_size, activation="relu", padding="same")(x1)
            x1 = layers.MaxPooling1D(2)(x1)
        x1 = layers.GlobalMaxPooling1D()(x1)
        
        # HTML Input branch
        html_input = layers.Input(shape=(self.html_max_length,), name="html_input")
        x2 = layers.Embedding(self.vocab_size, self.embedding_dim)(html_input)
        for f in self.filters:
            x2 = layers.Conv1D(f, self.kernel_size, activation="relu", padding="same")(x2)
            x2 = layers.MaxPooling1D(2)(x2)
        x2 = layers.GlobalMaxPooling1D()(x2)
        
        # Merge branches
        merged = layers.Concatenate()([x1, x2])
        merged = layers.Dropout(self.dropout_rate)(merged)
        merged = layers.Dense(64, activation="relu")(merged)
        output = layers.Dense(1, activation="sigmoid")(merged)
        
        model = models.Model(inputs=[url_input, html_input], outputs=output)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y):
        """Train the CNN model."""
        # X is expected to be a list [url_features, html_features]
        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs", 10)
        
        self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        self.trained = True

    def predict(self, X):
        """Make predictions."""
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        probs = self.model.predict(X)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        probs = self.model.predict(X)
        # Keras predict returns (n_samples, 1), scikit-learn predict_proba returns (n_samples, 2)
        return np.hstack([1-probs, probs])

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str):
        """Load a saved model."""
        self.model = models.load_model(path)
        self.trained = True

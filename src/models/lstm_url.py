"""LSTM URL-only phishing detection model."""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from .base import BaseModel

class LSTM_URL(BaseModel):
    """LSTM model for phishing detection using URL sequences only."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.embedding_dim = config.get("embedding_dim", 128)
        self.lstm_units = config.get("lstm_units", 64)
        self.dropout_rate = config.get("dropout", 0.3)
        self.max_seq_length = config.get("max_seq_length", 256)
        self.vocab_size = config.get("vocab_size", 128)
        
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.max_seq_length,), name="url_input")
        x = layers.Embedding(self.vocab_size, self.embedding_dim, name="embedding")(inputs)
        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=False, name="lstm"))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation="relu", name="fc")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        """Train the LSTM model."""
        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs", 10)
        validation_split = self.config.get("validation_split", 0.2)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
        
        self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
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
        return np.hstack([1-probs, probs])

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str):
        """Load a saved model."""
        self.model = models.load_model(path)
        self.trained = True

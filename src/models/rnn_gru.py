"""RNN-GRU URL phishing detection model."""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from .base import BaseModel

class RNN_GRU(BaseModel):
    """RNN-GRU model for phishing detection using URL sequences."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.embedding_dim = config.get("embedding_dim", 128)
        self.gru_units = config.get("gru_units", 64)
        self.dropout_rate = config.get("dropout", 0.3)
        self.max_seq_length = config.get("max_seq_length", 200)
        self.vocab_size = config.get("vocab_size", 100)
        
        self.model = self._build_model()

    def _build_model(self):
        # Input is now integer sequences: (max_seq_length,)
        inputs = layers.Input(shape=(self.max_seq_length,), name="url_input_seq")
        
        # Dense trainable Embedding layer
        x = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)(inputs)
        
        # 2-layer GRU
        num_layers = self.config.get("layers", 2)
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            x = layers.Bidirectional(layers.GRU(self.gru_units, return_sequences=return_sequences, name=f"gru_{i}"))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
        x = layers.Dense(64, activation="relu", name="fc")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get("learning_rate", 0.001)),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        """Train the RNN-GRU model."""
        batch_size = self.config.get("batch_size", 128)
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

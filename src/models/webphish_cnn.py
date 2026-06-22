"""WebPhish CNN phishing detection model."""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from .base import BaseModel

class WebPhish_CNN(BaseModel):
    """
    Multi-modal CNN model for phishing detection based on the WebPhish paper.
    
    Architecture:
    1. Input: URL chars (180) and HTML words (2000)
    2. Embedding: 16-dim for both
    3. Concatenation: (2180 x 16)
    4. 1D CNN: 32 filters, kernel size 8, ReLU
    5. Max Pooling
    6. Two Fully Connected layers with ReLU
    7. Output: Sigmoid
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.url_max_length = config.get("url_max_length", 180)
        self.html_max_length = config.get("html_max_length", 2000)
        self.url_vocab_size = config.get("url_vocab_size", 130)
        self.html_vocab_size = config.get("html_vocab_size", 50000)
        self.embedding_dim = config.get("embedding_dim", 16)
        self.filters = config.get("filters", 32)
        self.kernel_size = config.get("kernel_size", 8)
        self.dropout_rate = config.get("dropout", 0.2)
        
        self.model = self._build_model()

    def _build_model(self):
        # Input Layer
        url_input = layers.Input(shape=(self.url_max_length,), name="url_input")
        html_input = layers.Input(shape=(self.html_max_length,), name="html_input")
        
        # Embedding Layer
        url_emb = layers.Embedding(self.url_vocab_size, self.embedding_dim, name="url_emb")(url_input)
        html_emb = layers.Embedding(self.html_vocab_size, self.embedding_dim, name="html_emb")(html_input)
        
        # Concatenation Layer (None, 2180, 16)
        merged = layers.Concatenate(axis=1, name="concat_layer")([url_emb, html_emb])
        
        # Convolutional Neural Network (CNN)
        x = layers.Conv1D(self.filters, self.kernel_size, activation="relu", name="cnn_layer")(merged)
        
        # Max Pooling Layer
        # Using GlobalMaxPooling1D to extract the most critical features as per paper's intent
        x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
        
        # Fully Connected (FC) Layers
        x = layers.Dense(128, activation="relu", name="fc1")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation="relu", name="fc2")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output Layer
        output = layers.Dense(1, activation="sigmoid", name="output")(x)
        
        model = models.Model(inputs=[url_input, html_input], outputs=output)
        
        # Optimizer and Compile
        # Paper implementation reached 98.1% accuracy, likely using Adam with standard settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        """Train the CNN model."""
        # X is expected to be a list [url_features, html_features]
        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs", 20)
        validation_split = self.config.get("validation_split", 0.2)
        
        # Callbacks for better training
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
        self.model.save(f"{path}.keras")

    def load(self, path: str):
        """Load a saved model."""
        self.model = models.load_model(f"{path}.keras")
        self.trained = True

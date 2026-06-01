"""Deep Neural Network relying on structural features."""

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from .base import BaseModel

class Structural_DNN(BaseModel):
    """A DNN designed to process 1D structural numerical features."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.input_dim = config.get("input_dim", 15) # 15 structural features
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.model = None

    def build_model(self):
        """Build the DNN architecture."""
        model = Sequential([
            Input(shape=(self.input_dim,)),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        """Train the model."""
        if self.model is None:
            # Dynamically set input dimension based on feature processor output
            self.input_dim = X.shape[1]
            self.model = self.build_model()
            
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

    def predict(self, X):
        """Predict class labels."""
        y_prob = self.model.predict(X, batch_size=self.batch_size)
        return (y_prob > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict(X, batch_size=self.batch_size).flatten()

    def save(self, filepath: str):
        """Save Keras model."""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load Keras model."""
        self.model = tf.keras.models.load_model(filepath)

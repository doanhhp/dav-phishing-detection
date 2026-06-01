"""Anomaly Detection Autoencoder Model."""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

class AnomalyAutoencoder:
    def __init__(self, input_dim: int, config: dict = None):
        self.input_dim = input_dim
        self.config = config or {}
        self.model = self._build_model()
        
    def _build_model(self):
        # Hyperparameters
        encoding_dim = self.config.get('encoding_dim', 8)
        hidden_dim = self.config.get('hidden_dim', 16)
        dropout_rate = self.config.get('dropout_rate', 0.1)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        encoded = Dense(hidden_dim, activation='relu')(inputs)
        encoded = Dropout(dropout_rate)(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(hidden_dim, activation='relu')(encoded)
        decoded = Dropout(dropout_rate)(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        # Autoencoder
        autoencoder = Model(inputs, decoded)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mse')
        
        return autoencoder

    def fit(self, X_train, epochs=20, batch_size=32, validation_split=0.1):
        """Train the autoencoder."""
        return self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1
        )
        
    def get_reconstruction_error(self, X):
        """Calculate MSE between input and reconstruction."""
        reconstructions = self.model.predict(X)
        mse = tf.reduce_mean(tf.square(X - reconstructions), axis=1)
        return mse.numpy()

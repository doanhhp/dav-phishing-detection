"""EGSO-CNN phishing detection model (2025)."""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from .base import BaseModel

# Reparameterization trick for VAE
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a feature."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(layers.Layer):
    """Custom layer to calculate and add VAE loss."""
    def call(self, inputs):
        original, reconstructed, z_mean, z_log_var = inputs
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.binary_crossentropy(original, reconstructed), axis=-1)
        )
        # KL divergence loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        # Scale VAE loss
        vae_loss = 0.1 * (reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return reconstructed

class EGSO_CNN(BaseModel):
    """Optimized CNN model with VAE and feature reduction."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_features = config.get("features", {}).get("tfidf_vae", {}).get("max_features", 5000)
        self.latent_dim = config.get("svd_components", 100) # Use this as VAE latent dim
        self.filters = config.get("filters", [64, 128, 256])
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout_rate = config.get("dropout", 0.5)
        
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.max_features,), name="tfidf_input")
        
        # --- VAE Encoder ---
        x_enc = layers.Dense(512, activation="relu")(inputs)
        x_enc = layers.Dense(256, activation="relu")(x_enc)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x_enc)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x_enc)
        z = Sampling()([z_mean, z_log_var])
        
        # --- VAE Decoder ---
        x_dec = layers.Dense(256, activation="relu")(z)
        x_dec = layers.Dense(512, activation="relu")(x_dec)
        reconstructed_raw = layers.Dense(self.max_features, activation="sigmoid")(x_dec)
        
        # Apply VAE loss layer
        reconstructed = VAELossLayer(name="vae_output")([inputs, reconstructed_raw, z_mean, z_log_var])
        
        # --- Concatenate Original + Latent Features ---
        combined_features = layers.Concatenate()([inputs, z])
        combined_dim = self.max_features + self.latent_dim
        
        # Reshape for Conv1D
        x_cnn = layers.Reshape((combined_dim, 1))(combined_features)
        
        # --- CNN Pipeline ---
        for f in self.filters:
            x_cnn = layers.Conv1D(f, self.kernel_size, activation="relu", padding="same")(x_cnn)
            x_cnn = layers.MaxPooling1D(pool_size=2)(x_cnn)
            
        x_cnn = layers.GlobalMaxPooling1D()(x_cnn)
        x_cnn = layers.Dropout(self.dropout_rate)(x_cnn)
        x_cnn = layers.Dense(128, activation="relu")(x_cnn)
        x_cnn = layers.Dropout(self.dropout_rate)(x_cnn)
        
        outputs = layers.Dense(1, activation="sigmoid", name="classification_output")(x_cnn)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get("learning_rate", 0.001)),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        """Train the EGSO-CNN model."""
        batch_size = self.config.get("batch_size", 128)
        epochs = self.config.get("epochs", 100)
        validation_split = self.config.get("validation_split", 0.2)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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

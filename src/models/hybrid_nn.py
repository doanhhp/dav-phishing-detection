import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from .base import BaseModel

class HybridNN(BaseModel):
    """
    Hybrid Neural Network that combines Deep Learning on text (URL + HTML)
    with Tabular Structural features.
    
    Architecture:
    Branch 1: URL chars -> Embedding -> Conv1D -> GlobalMaxPooling1D
    Branch 2: HTML words -> Embedding -> Conv1D -> GlobalMaxPooling1D
    Branch 3: Structural features (15 dims) -> Dense
    Fusion: Concatenate all branches -> Dense -> Output
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
        self.dense_units = config.get("dense_units", 64)
        
        self.model = self._build_model()

    def _build_model(self):
        # 1. URL Branch
        url_input = layers.Input(shape=(self.url_max_length,), name="url_input")
        url_emb = layers.Embedding(self.url_vocab_size, self.embedding_dim, name="url_emb")(url_input)
        url_cnn = layers.Conv1D(self.filters, self.kernel_size, activation="relu")(url_emb)
        url_pool = layers.GlobalMaxPooling1D()(url_cnn)
        
        # 2. HTML Branch
        html_input = layers.Input(shape=(self.html_max_length,), name="html_input")
        html_emb = layers.Embedding(self.html_vocab_size, self.embedding_dim, name="html_emb")(html_input)
        html_cnn = layers.Conv1D(self.filters, self.kernel_size, activation="relu")(html_emb)
        html_pool = layers.GlobalMaxPooling1D()(html_cnn)
        
        # 3. Structural Branch
        struct_input = layers.Input(shape=(15,), name="struct_input")
        struct_dense = layers.Dense(16, activation="relu")(struct_input)
        
        # 4. Fusion
        merged = layers.Concatenate()([url_pool, html_pool, struct_dense])
        
        # FC Layers
        x = layers.Dense(self.dense_units, activation="relu")(merged)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.dense_units // 2, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output Layer
        output = layers.Dense(1, activation="sigmoid", name="output")(x)
        
        model = models.Model(inputs=[url_input, html_input, struct_input], outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get("learning_rate", 0.001)),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        # X is expected to be a list [url_features, html_features, struct_features]
        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 15)
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
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        probs = self.model.predict(X)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        probs = self.model.predict(X)
        return np.hstack([1-probs, probs])

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = models.load_model(path)
        self.trained = True

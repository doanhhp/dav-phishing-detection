"""RNN-GRU feature processor for URL sequences."""

import numpy as np
import pandas as pd
import tensorflow as tf
import re

class RnnGruProcessor:
    """Processes URL sequences into integer tokens for Embedding layers."""

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.max_seq_length = config.get("max_seq_length", 200)
        self.vocab_size = config.get("vocab_size", 1000) # Increased for subwords
        
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_sequence_length=self.max_seq_length,
            split='whitespace',
            standardize='lower_and_strip_punctuation'
        )

    def _extract_text(self, X):
        if isinstance(X, pd.Series):
            urls = X.values.astype(str)
        elif isinstance(X, pd.DataFrame):
            urls = X.iloc[:, 0].values.astype(str)
        else:
            urls = np.array(X).astype(str)
            
        # Add spaces around punctuation to treat them as separate words/subwords
        processed = [re.sub(r'([./\-_?=&])', r' \1 ', u) for u in urls]
        return np.array(processed)

    def fit_transform(self, X, y=None):
        """Fit vocabulary and transform sequences to integer matrices."""
        text_data = self._extract_text(X)
        self.vectorizer.adapt(text_data)
        self.fitted = True
        return self.vectorizer(text_data).numpy()

    def transform(self, X):
        """Transform sequences into integer matrices."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        text_data = self._extract_text(X)
        return self.vectorizer(text_data).numpy()

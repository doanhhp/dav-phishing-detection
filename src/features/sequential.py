"""Sequential token processor for LSTM models."""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SequentialTokenProcessor:
    """Processes character sequences for LSTM URL-Only model."""

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.max_seq_length = config.get("max_seq_length", 256)
        self.vocab_size = config.get("vocab_size", 128)
        self.tokenizer = Tokenizer(num_words=self.vocab_size, char_level=True, oov_token="<OOV>")

    def _extract_urls(self, X):
        if isinstance(X, pd.Series):
            return X.values.astype(str)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].values.astype(str)
        else:
            return np.array(X).astype(str)

    def fit_transform(self, X, y=None):
        """Fit tokenizer and transform sequences."""
        urls = self._extract_urls(X)
        self.tokenizer.fit_on_texts(urls)
        self.fitted = True
        return self.transform(X)

    def transform(self, X):
        """Transform sequences."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        urls = self._extract_urls(X)
        sequences = self.tokenizer.texts_to_sequences(urls)
        padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post', truncating='post')
        return padded

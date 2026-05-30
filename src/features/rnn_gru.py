"""RNN-GRU feature processor for URL sequences."""

import numpy as np
import pandas as pd
from collections import Counter

class RnnGruProcessor:
    """Processes character sequences into one-hot encoded matrices (200x100)."""

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.max_seq_length = config.get("max_seq_length", 200)
        self.vocab_size = config.get("vocab_size", 100)
        self.char_to_idx = {}

    def _extract_urls(self, X):
        if isinstance(X, pd.Series):
            return X.values.astype(str)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].values.astype(str)
        else:
            return np.array(X).astype(str)

    def fit_transform(self, X, y=None):
        """Fit vocabulary and transform sequences to one-hot matrices."""
        urls = self._extract_urls(X)
        
        # Build dictionary of most common ASCII characters
        char_counter = Counter()
        for url in urls:
            char_counter.update(url)
            
        most_common = char_counter.most_common(self.vocab_size)
        self.char_to_idx = {char: idx for idx, (char, _) in enumerate(most_common)}
        
        self.fitted = True
        return self.transform(X)

    def transform(self, X):
        """Transform sequences into one-hot matrices."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        urls = self._extract_urls(X)
        
        num_samples = len(urls)
        # Initialize matrix with zeros: (samples, max_seq_length, vocab_size)
        X_encoded = np.zeros((num_samples, self.max_seq_length, self.vocab_size), dtype=np.float32)
        
        for i, url in enumerate(urls):
            # Cap at max_seq_length
            url_chars = url[:self.max_seq_length]
            for j, char in enumerate(url_chars):
                if char in self.char_to_idx:
                    idx = self.char_to_idx[char]
                    X_encoded[i, j, idx] = 1.0
                    
        return X_encoded

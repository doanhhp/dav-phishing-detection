"""Multimodal feature processor for WebPhish CNN models."""

import numpy as np
import pandas as pd

class MultimodalProcessor:
    """Processes URL and HTML features for WebPhish CNN model."""

    def __init__(self, config: dict):
        self.config = config
        self.url_max_length = config.get("url_max_length", 200)
        self.html_max_length = config.get("html_max_length", 5000)
        self.fitted = False
        
        # Simple character mapping (0 for padding, 1 for OOV)
        self.char_map = {chr(i): i + 2 for i in range(128)} # Basic ASCII
        self.vocab_size = 130

    def _tokenize(self, text, max_length):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        tokens = [self.char_map.get(c, 1) for c in text[:max_length]]
        # Padding
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        return tokens

    def fit_transform(self, X, y=None):
        """Fit and transform multimodal features."""
        self.fitted = True
        return self.transform(X)

    def transform(self, X):
        """Transform multimodal features."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            # Expecting columns 'url' and 'html'
            urls = X['url'].values if 'url' in X.columns else X.iloc[:, 0].values
            htmls = X['html'].values if 'html' in X.columns else X.iloc[:, 1].values
        else:
            # Fallback if X is a list of tuples or similar
            urls = [item[0] for item in X]
            htmls = [item[1] for item in X]
            
        url_features = np.array([self._tokenize(url, self.url_max_length) for url in urls])
        html_features = np.array([self._tokenize(html, self.html_max_length) for html in htmls])
        
        return [url_features, html_features]

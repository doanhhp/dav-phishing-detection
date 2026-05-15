"""Multimodal feature processor for WebPhish CNN models."""

import re
import numpy as np
import pandas as pd
from collections import Counter

class MultimodalProcessor:
    """Processes URL (char-level) and HTML (word-level) features for WebPhish CNN."""

    def __init__(self, config: dict):
        self.config = config
        self.url_max_length = config.get("url_max_length", 180)
        self.html_max_length = config.get("html_max_length", 2000)
        self.html_vocab_size = config.get("html_vocab_size", 50000)
        self.fitted = False
        
        # URL character mapping (0 for padding, 1 for OOV)
        self.char_map = {chr(i): i + 2 for i in range(128)} 
        self.url_actual_vocab_size = 130
        
        # HTML word mapping
        self.word_map = {}
        self.html_actual_vocab_size = 0

    def _split_html(self, html):
        """Splits HTML into words and punctuation."""
        if not isinstance(html, str):
            html = str(html) if html is not None else ""
        # Split by words and keep punctuation as separate tokens
        return re.findall(r"[\w']+|[^\w\s]", html.lower())

    def _tokenize_url(self, url):
        if not isinstance(url, str):
            url = str(url) if url is not None else ""
        tokens = [self.char_map.get(c, 1) for c in url[:self.url_max_length]]
        if len(tokens) < self.url_max_length:
            tokens.extend([0] * (self.url_max_length - len(tokens)))
        return tokens

    def _tokenize_html(self, html):
        tokens = self._split_html(html)
        # 0: pad, 1: OOV, 2+: words
        token_ids = [self.word_map.get(t, 1) for t in tokens[:self.html_max_length]]
        if len(token_ids) < self.html_max_length:
            token_ids.extend([0] * (self.html_max_length - len(token_ids)))
        return token_ids

    def fit(self, X, y=None):
        """Build HTML vocabulary from data."""
        if isinstance(X, pd.DataFrame):
            htmls = X['html'].values if 'html' in X.columns else X.iloc[:, 1].values
        else:
            htmls = [item[1] for item in X]
            
        word_counter = Counter()
        for html in htmls:
            word_counter.update(self._split_html(html))
            
        # Limit vocabulary size
        most_common = word_counter.most_common(self.html_vocab_size - 2)
        self.word_map = {word: i + 2 for i, (word, _) in enumerate(most_common)}
        self.html_actual_vocab_size = len(self.word_map) + 2
        
        self.fitted = True
        return self

    def fit_transform(self, X, y=None):
        """Fit and transform multimodal features."""
        return self.fit(X, y).transform(X)

    def transform(self, X):
        """Transform multimodal features."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            urls = X['url'].values if 'url' in X.columns else X.iloc[:, 0].values
            htmls = X['html'].values if 'html' in X.columns else X.iloc[:, 1].values
        else:
            urls = [item[0] for item in X]
            htmls = [item[1] for item in X]
            
        url_features = np.array([self._tokenize_url(url) for url in urls])
        html_features = np.array([self._tokenize_html(html) for html in htmls])
        
        return [url_features, html_features]

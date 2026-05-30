"""TF-IDF and StandardScaler feature processor for EGSO-CNN models."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class TfidfVaeProcessor:
    """Processes features using TF-IDF and StandardScaler."""

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.max_features = config.get("max_features", 5000)
        
        self.tfidf = TfidfVectorizer(max_features=self.max_features, analyzer='char_wb', ngram_range=(1, 4))
        # StandardScaler expects dense array, we use with_mean=False if tfidf outputs sparse
        # but to strictly follow standard scaler (mean=0, std=1) we need dense data.
        self.scaler = StandardScaler()
        
    def _extract_urls(self, X):
        if isinstance(X, pd.Series):
            return X.values.astype(str)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].values.astype(str)
        else:
            return np.array(X).astype(str)

    def fit_transform(self, X, y=None):
        """Fit TF-IDF and Scaler, then transform features."""
        urls = self._extract_urls(X)
        tfidf_features = self.tfidf.fit_transform(urls).toarray()
        scaled_features = self.scaler.fit_transform(tfidf_features)
        self.fitted = True
        return scaled_features

    def transform(self, X):
        """Transform features using fitted TF-IDF and Scaler."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        urls = self._extract_urls(X)
        tfidf_features = self.tfidf.transform(urls).toarray()
        return self.scaler.transform(tfidf_features)

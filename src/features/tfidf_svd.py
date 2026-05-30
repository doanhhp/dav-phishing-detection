"""TF-IDF + SVD feature processor for EGSO-CNN models."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

class TfidfSvdProcessor:
    """Processes features using TF-IDF and SVD dimensionality reduction."""

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.max_features = config.get("max_features", 5000)
        self.svd_components = config.get("svd_components", 100)
        
        self.tfidf = TfidfVectorizer(max_features=self.max_features, analyzer='char_wb', ngram_range=(1, 4))
        self.svd = TruncatedSVD(n_components=self.svd_components, random_state=42)
        self.pipeline = Pipeline([('tfidf', self.tfidf), ('svd', self.svd)])

    def _extract_urls(self, X):
        if isinstance(X, pd.Series):
            return X.values.astype(str)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].values.astype(str)
        else:
            return np.array(X).astype(str)

    def fit_transform(self, X, y=None):
        """Fit TF-IDF and SVD, then transform features."""
        urls = self._extract_urls(X)
        features = self.pipeline.fit_transform(urls)
        self.fitted = True
        return features

    def transform(self, X):
        """Transform features using fitted TF-IDF and SVD."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        urls = self._extract_urls(X)
        return self.pipeline.transform(urls)

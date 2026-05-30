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
        
    def _extract_text(self, X):
        from urllib.parse import urlparse
        from bs4 import BeautifulSoup
        import traceback

        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            urls = X.iloc[:, 0].fillna("").astype(str).values
            htmls = X.iloc[:, 1].fillna("").astype(str).values
            
            parsed_texts = []
            for u, h in zip(urls, htmls):
                # 1. Parse URL components
                try:
                    u_to_parse = u if u.startswith("http") else "http://" + u
                    parsed = urlparse(u_to_parse)
                    url_features = f"{parsed.scheme} {parsed.netloc} {parsed.path} {parsed.query}".replace(".", " ").replace("/", " ")
                except Exception:
                    url_features = u.replace(".", " ").replace("/", " ")
                
                # 2. Parse HTML components
                html_features = ""
                try:
                    if h and len(h) > 5:
                        soup = BeautifulSoup(h, "html.parser")
                        links = " ".join([a.get_text(separator=' ', strip=True) for a in soup.find_all('a')])
                        lists = " ".join([li.get_text(separator=' ', strip=True) for li in soup.find_all(['ul', 'ol', 'li'])])
                        titles = " ".join([t.get_text(separator=' ', strip=True) for t in soup.find_all('title')])
                        html_features = f"{titles} {links} {lists}"
                except Exception:
                    pass
                
                parsed_texts.append(f"{url_features} {html_features}")
            return parsed_texts
        elif isinstance(X, pd.Series):
            return X.fillna("").astype(str).values
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna("").astype(str).values
        else:
            return np.array(X).astype(str)

    def fit_transform(self, X, y=None):
        """Fit TF-IDF and Scaler, then transform features."""
        text_data = self._extract_text(X)
        tfidf_features = self.tfidf.fit_transform(text_data).toarray()
        scaled_features = self.scaler.fit_transform(tfidf_features)
        self.fitted = True
        return scaled_features

    def transform(self, X):
        """Transform features using fitted TF-IDF and Scaler."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        text_data = self._extract_text(X)
        tfidf_features = self.tfidf.transform(text_data).toarray()
        return self.scaler.transform(tfidf_features)

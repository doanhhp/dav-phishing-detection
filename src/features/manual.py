"""Manual feature processor for SVM+KNN models."""

import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse

class ManualFeatureProcessor:
    """Processes manual features for hybrid SVM+KNN model."""

    def __init__(self, config: dict):
        self.config = config
        self.n_features = config.get("n_features", 48)
        self.fitted = False

    def _extract_url_features(self, url):
        """Extract numerical features from a URL."""
        if not isinstance(url, str):
            url = str(url)
            
        features = []
        
        # 1. URL Length
        features.append(len(url))
        
        # 2. Number of dots
        features.append(url.count('.'))
        
        # 3. Number of hyphens
        features.append(url.count('-'))
        
        # 4. Number of underscores
        features.append(url.count('_'))
        
        # 5. Number of slash
        features.append(url.count('/'))
        
        # 6. Number of question marks
        features.append(url.count('?'))
        
        # 7. Number of equals
        features.append(url.count('='))
        
        # 8. Number of ampersands
        features.append(url.count('&'))
        
        # 9. Number of digits
        features.append(sum(c.isdigit() for c in url))
        
        # 10. Number of letters
        features.append(sum(c.isalpha() for c in url))
        
        # 11. Is IP address
        ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
        features.append(1 if re.search(ip_pattern, url) else 0)
        
        # 12. Has @ symbol
        features.append(1 if '@' in url else 0)
        
        # 13. Has double slash (redirection)
        features.append(1 if url.rfind('//') > 7 else 0)
        
        # 14. URL depth
        parsed = urlparse(url)
        features.append(len([p for p in parsed.path.split('/') if p]))
        
        # 15. HTTPS presence
        features.append(1 if parsed.scheme == 'https' else 0)
        
        # 16. Number of subdomains
        domain = parsed.netloc
        features.append(len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0)

        # Fill remaining to reach n_features (if needed) with 0s or more features
        while len(features) < self.n_features:
            features.append(0)
            
        return features[:self.n_features]

    def fit_transform(self, X, y=None):
        """Fit and transform features."""
        self.fitted = True
        return self.transform(X)

    def transform(self, X):
        """Transform features."""
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
        
        if isinstance(X, pd.Series):
            urls = X.values
        elif isinstance(X, pd.DataFrame):
            # Assume first column is URL
            urls = X.iloc[:, 0].values
        else:
            urls = X
            
        feature_matrix = [self._extract_url_features(url) for url in urls]
        return np.array(feature_matrix)

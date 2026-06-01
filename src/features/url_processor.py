"""URL-only feature processor for System 1 (Human Intuition) Cascade."""

import numpy as np
import pandas as pd
import math
import re
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler

class UrlProcessor:
    """Extracts enriched human-intuition URL features for Fast-Slow cascading."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.fitted = False

        self.suspicious_tlds = {
            '.xyz', '.top', '.pw', '.cc', '.tk', '.ml', '.ga', '.cf', '.gq', '.icu',
            '.vip', '.work', '.click', '.link', '.club', '.online', '.site', '.biz'
        }
        
        self.top_brands = [
            'paypal', 'microsoft', 'apple', 'amazon', 'facebook', 'chase', 
            'netflix', 'google', 'wellsfargo', 'bankofamerica', 'whatsapp', 
            'instagram', 'linkedin', 'dhl', 'fedex', 'ups', 'adobe', 'dropbox', 
            'yahoo', 'binance', 'coinbase'
        ]
        
        self.phish_keywords = [
            'login', 'secure', 'verify', 'account', 'auth', 'update', 
            'support', 'service', 'webscr', 'confirm', 'billing', 'recover'
        ]

    def _entropy(self, string):
        """Calculates the Shannon entropy of a string."""
        if not string: return 0.0
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        return -sum([p * math.log(p) / math.log(2.0) for p in prob])

    def _extract_url_features(self, url):
        features = []
        url_lower = url.lower()
        if not url_lower.startswith('http'):
            url_lower = f"http://{url_lower}"
            
        parsed = urlparse(url_lower)
        domain = parsed.netloc
        path = parsed.path
        
        # 1. Base Lexical Features
        features.append(len(url))
        features.append(url.count('.'))
        features.append(sum(url.count(c) for c in ['-', '@', '?', '=', '%', '_']))
        features.append(sum(c.isdigit() for c in url) / max(1, len(url)))
        features.append(max(0, len(domain.split('.')) - 2)) # subdomains
        features.append(path.count('/'))
        features.append(1 if '-' in domain else 0)
        
        # 2. Domain Entropy (Randomness)
        features.append(self._entropy(domain))
        
        # 3. Suspicious TLD
        tld = '.' + domain.split('.')[-1] if '.' in domain else ''
        features.append(1 if tld in self.suspicious_tlds else 0)
        
        # 4. Brand Impersonation (Brand in subdomain or path, but not root domain)
        root_domain_parts = domain.split('.')[-2:] if len(domain.split('.')) >= 2 else [domain]
        root_domain = '.'.join(root_domain_parts)
        
        brand_impersonation = 0
        for brand in self.top_brands:
            # If brand is in the URL but NOT the root domain (it's in subdomain or path)
            if brand in url_lower and brand not in root_domain:
                brand_impersonation = 1
                break
        features.append(brand_impersonation)
        
        # 5. Keyword Stuffing Score
        keyword_score = sum(url_lower.count(kw) for kw in self.phish_keywords)
        features.append(keyword_score)
        
        return features

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame) and 'Data' in X.columns:
            urls = X['Data'].astype(str).values
        elif isinstance(X, pd.Series):
            urls = X.astype(str).values
        else:
            raise ValueError("UrlProcessor expects a Series or DataFrame with 'Data' column.")
        
        num_features = []
        for u in urls:
            num_features.append(self._extract_url_features(u))
            
        scaled_features = self.scaler.fit_transform(np.array(num_features))
        self.fitted = True
        return scaled_features

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
            
        if isinstance(X, pd.DataFrame) and 'Data' in X.columns:
            urls = X['Data'].astype(str).values
        elif isinstance(X, pd.Series):
            urls = X.astype(str).values
        else:
            raise ValueError("UrlProcessor expects a Series or DataFrame with 'Data' column.")
        
        num_features = []
        for u in urls:
            num_features.append(self._extract_url_features(u))
            
        return self.scaler.transform(np.array(num_features))

    def get_feature_names(self):
        return [
            'url_length', 'url_num_dots', 'url_num_special_chars', 'url_digit_ratio',
            'url_num_subdomains', 'url_path_depth', 'url_hyphen_domain',
            'domain_entropy', 'tld_suspiciousness', 'brand_impersonation', 'keyword_stuffing_score'
        ]

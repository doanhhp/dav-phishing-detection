"""Structural feature processor for URL and HTML."""

import numpy as np
import pandas as pd
import re
import math
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib

class StructuralProcessor:
    """Extracts structural features from URLs and HTML for robust classification."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        # Structural Skeletoning (DOM Tag NLP - Template Mining)
        self.tfidf = TfidfVectorizer(max_features=50, ngram_range=(2, 3), token_pattern=r'(?u)\b\w+\b')
        # XPath N-Grams (DOM Hierarchy Sequences)
        self.xpath_tfidf = TfidfVectorizer(max_features=25, ngram_range=(2, 3), token_pattern=r'(?u)\S+')
        
        # Spatial Zone TF-IDF (Top, Middle, Bottom)
        self.zone1_tfidf = TfidfVectorizer(max_features=20, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
        self.zone2_tfidf = TfidfVectorizer(max_features=20, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
        self.zone3_tfidf = TfidfVectorizer(max_features=20, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
        
        self.fitted = False

    def _entropy(self, string):
        """Calculates the Shannon entropy of a string"""
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    def _extract_numerical_features(self, url, html, domain):
        """Extract numerical structural features from URL and HTML."""
        features = []
        html_lower = html.lower()
        
        # --- URL Features (Lexical) ---
        features.append(len(url))
        features.append(url.count('.'))
        features.append(1 if str(url).lower().startswith('https') else 0)
        features.append(sum(url.count(c) for c in ['-', '@', '?', '=', '%', '_']))
        features.append(sum(c.isdigit() for c in url) / max(1, len(url)))
        parsed = urlparse(url if url.startswith('http') else f"http://{url}")
        features.append(max(0, len(domain.split('.')) - 2))
        features.append(self._entropy(url) if url else 0)
        features.append(parsed.path.count('/'))
        login_keywords = ['login', 'signin', 'auth', 'secure', 'update', 'account', 'verify', 'webscr']
        features.append(1 if any(kw in url.lower() for kw in login_keywords) else 0)
        features.append(1 if '-' in domain else 0)
        
        # --- Brand Mimicry removed per user request ---
        
        # --- HTML Features ---
        features.append(len(html))
        
        tags = re.findall(r'<[^>]+>', html)
        features.append(len(tags))
        
        text_content = re.sub(r'<[^>]+>', '', html)
        features.append(len(text_content) / max(1, len(html)))
        
        features.append(html_lower.count('<script'))
        # html_input_form_count removed as it's redundant with input_to_p_ratio
        
        links = re.findall(r'href=[\'"]?([^\'" >]+)', html_lower)
        external_links = sum(1 for link in links if link.startswith('http') and domain not in link)
        total_links = max(1, len(links))
        features.append(external_links / total_links)
        
        features.append(len(re.findall(r'<input[^>]+type=[\'"]?password[\'"]?', html_lower)))
        
        # Dead Link Ratio
        empty_links = len(re.findall(r'<a[^>]+href=[\'"]?(#|javascript:void\\(0\\)|)[\'"]?', html_lower))
        features.append(empty_links / total_links)
        
        # Input-to-Content Ratio
        input_btn_count = html_lower.count('<input') + html_lower.count('<button')
        p_count = max(1, html_lower.count('<p>'))
        features.append(input_btn_count / p_count)
        
        # CSS Anomaly Detection
        css_hidden = len(re.findall(r'display:\s*none|visibility:\s*hidden|opacity:\s*0', html_lower))
        features.append(css_hidden)
        # css_zindex removed as it showed very low importance
        
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        features.append(len(title_match.group(1).strip()) if title_match else 0)
        
        # Brand Discrepancy (NLP)
        title_text = title_match.group(1).strip().lower() if title_match else ""
        for stop in ['login', 'sign in', 'secure', 'update', 'account', 'home', 'welcome']:
            title_text = title_text.replace(stop, '')
        title_text = re.sub(r'[^a-z0-9]', '', title_text)
        
        domain_alphanum = re.sub(r'[^a-z0-9]', '', domain)
        if not title_text:
            brand_discrepancy = 0.0
        else:
            similarity = difflib.SequenceMatcher(None, title_text, domain_alphanum).ratio()
            if len(title_text) > 3 and title_text in domain_alphanum:
                similarity = 1.0
            brand_discrepancy = 1.0 - similarity
        features.append(brand_discrepancy)
        
        # --- Advanced EDA & Positional Features ---
        try:
            import lxml.html
            tree = lxml.html.fromstring(html)
            
            all_tags_list = list(tree.iter())
            
            def get_max_depth(element, current_depth):
                m_depth = current_depth
                for child in element.iterchildren():
                    if isinstance(child.tag, str):
                        m_depth = max(m_depth, get_max_depth(child, current_depth + 1))
                return m_depth
                
            features.append(get_max_depth(tree, 0))
            
            all_tags = [tag.tag for tag in all_tags_list if isinstance(tag.tag, str)]
            features.append(len(set(all_tags)))
            
            resources = [t for t in all_tags_list if isinstance(t.tag, str) and t.tag in ['img', 'script', 'link']]
            if resources:
                ext_res = 0
                for tag in resources:
                    src = tag.get('src') or tag.get('href')
                    if src and str(src).startswith('http') and domain not in str(src):
                        ext_res += 1
                features.append(ext_res / len(resources))
            else:
                features.append(0)
                
            # Form Action Discrepancy
            foreign_form = 0
            for form in [t for t in all_tags_list if isinstance(t.tag, str) and t.tag == 'form']:
                action = form.get('action', '')
                if action and str(action).startswith('http'):
                    action_domain = urlparse(action).netloc.lower()
                    if action_domain != domain and not action_domain.endswith('.' + domain):
                        foreign_form = 1
                        break
            features.append(foreign_form)
            
            # --- Cloaking / Evasion Features ---
            scripts = [t for t in all_tags_list if isinstance(t.tag, str) and t.tag == 'script']
            js_len = sum(len(t.text_content()) for t in scripts if t.text_content())
            features.append(js_len / max(1, len(html))) # html_js_ratio
            
            body_tag_count = len([t for t in all_tags_list if isinstance(t.tag, str) and t.tag != 'html' and t.tag != 'head'])
            features.append(body_tag_count) # body_tag_count
            
            iframe_count = len([t for t in all_tags_list if isinstance(t.tag, str) and t.tag == 'iframe'])
            features.append(iframe_count) # iframe_count
                
        except Exception:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        return features

    def _extract_tag_sequence(self, html):
        """Extract sequence of HTML tags for TF-IDF (Structural Skeletoning)"""
        tags = re.findall(r'<([a-zA-Z0-9]+)[^>]*>', str(html).lower())
        return " ".join(tags)

    def _extract_spatial_zones(self, html):
        """Extract tags and split them into 3 spatial zones (Top, Middle, Bottom)"""
        tags = re.findall(r'<([a-zA-Z0-9]+)[^>]*>', str(html).lower())
        if not tags:
            return "", "", ""
            
        chunk_size = max(1, len(tags) // 3)
        zone1 = " ".join(tags[:chunk_size])
        zone2 = " ".join(tags[chunk_size:2*chunk_size])
        zone3 = " ".join(tags[2*chunk_size:])
        return zone1, zone2, zone3
        
    def _extract_xpath_sequence(self, html):
        """Extract sequence of XPath paths for all tags in the HTML (DOM Hierarchy)"""
        try:
            import lxml.html
            tree = lxml.html.fromstring(html)
            
            def build_paths(element, current_path):
                paths = []
                for child in element.iterchildren():
                    if isinstance(child.tag, str):
                        new_path = f"{current_path}/{child.tag}" if current_path else child.tag
                        paths.append(f"<{new_path}>")
                        paths.extend(build_paths(child, new_path))
                return paths
                
            return " ".join(build_paths(tree, ""))
        except Exception:
            return ""

    def fit_transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("StructuralProcessor expects a DataFrame with 'Data' and 'html' columns.")
            
        urls = X['Data'].astype(str).values
        htmls = X['html'].astype(str).values if 'html' in X.columns else [''] * len(X)
        
        # Extract numerical features
        num_features = []
        tag_sequences = []
        xpath_sequences = []
        z1_seqs, z2_seqs, z3_seqs = [], [], []
        
        for u, h in zip(urls, htmls):
            domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
            num_features.append(self._extract_numerical_features(u, h, domain))
            tag_sequences.append(self._extract_tag_sequence(h))
            xpath_sequences.append(self._extract_xpath_sequence(h))
            
            z1, z2, z3 = self._extract_spatial_zones(h)
            z1_seqs.append(z1)
            z2_seqs.append(z2)
            z3_seqs.append(z3)
            
        # Fit and transform TF-IDF on sequences
        tfidf_features = self.tfidf.fit_transform(tag_sequences).toarray()
        xpath_features = self.xpath_tfidf.fit_transform(xpath_sequences).toarray()
        
        z1_features = self.zone1_tfidf.fit_transform(z1_seqs).toarray()
        z2_features = self.zone2_tfidf.fit_transform(z2_seqs).toarray()
        z3_features = self.zone3_tfidf.fit_transform(z3_seqs).toarray()
        
        # Combine all features
        combined_features = np.hstack((
            np.array(num_features), 
            tfidf_features, 
            xpath_features,
            z1_features, z2_features, z3_features
        ))
        
        # Scale
        scaled_features = self.scaler.fit_transform(combined_features)
        self.fitted = True
        return scaled_features

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("StructuralProcessor expects a DataFrame with 'Data' and 'html' columns.")
            
        urls = X['Data'].astype(str).values
        htmls = X['html'].astype(str).values if 'html' in X.columns else [''] * len(X)
        
        num_features = []
        tag_sequences = []
        xpath_sequences = []
        z1_seqs, z2_seqs, z3_seqs = [], [], []
        
        for u, h in zip(urls, htmls):
            domain = urlparse(u if u.startswith('http') else f"http://{u}").netloc
            num_features.append(self._extract_numerical_features(u, h, domain))
            tag_sequences.append(self._extract_tag_sequence(h))
            xpath_sequences.append(self._extract_xpath_sequence(h))
            
            z1, z2, z3 = self._extract_spatial_zones(h)
            z1_seqs.append(z1)
            z2_seqs.append(z2)
            z3_seqs.append(z3)
            
        tfidf_features = self.tfidf.transform(tag_sequences).toarray()
        xpath_features = self.xpath_tfidf.transform(xpath_sequences).toarray()
        
        z1_features = self.zone1_tfidf.transform(z1_seqs).toarray()
        z2_features = self.zone2_tfidf.transform(z2_seqs).toarray()
        z3_features = self.zone3_tfidf.transform(z3_seqs).toarray()
        
        combined_features = np.hstack((
            np.array(num_features), 
            tfidf_features, 
            xpath_features,
            z1_features, z2_features, z3_features
        ))
        
        return self.scaler.transform(combined_features)

    def get_feature_names(self):
        base_features = [
            'url_length', 'url_num_dots', 'is_https', 'url_num_special_chars', 'url_digit_ratio',
            'url_num_subdomains', 'url_entropy', 'url_path_depth', 'url_has_login', 'url_hyphen_domain',
            'html_length', 'html_num_tags', 'html_text_ratio', 'html_script_count',
            'html_external_link_ratio', 'html_password_input_count',
            'html_empty_link_ratio', 'html_input_to_p_ratio', 'css_hidden_count',
            'html_title_length', 'brand_discrepancy', 'dom_depth', 'tag_diversity', 'external_resource_ratio', 'foreign_form_action',
            'html_js_ratio', 'body_tag_count', 'iframe_count'
        ]
        
        if getattr(self, 'fitted', False) and hasattr(self.tfidf, 'get_feature_names_out') and hasattr(self.xpath_tfidf, 'get_feature_names_out'):
            tfidf_features = [f"tag_tfidf_{feat}" for feat in self.tfidf.get_feature_names_out()]
            xpath_features = [f"xpath_tfidf_{feat}" for feat in self.xpath_tfidf.get_feature_names_out()]
            z1_features = [f"z1_tfidf_{feat}" for feat in self.zone1_tfidf.get_feature_names_out()]
            z2_features = [f"z2_tfidf_{feat}" for feat in self.zone2_tfidf.get_feature_names_out()]
            z3_features = [f"z3_tfidf_{feat}" for feat in self.zone3_tfidf.get_feature_names_out()]
            return base_features + tfidf_features + xpath_features + z1_features + z2_features + z3_features
        return base_features + [f"tag_tfidf_{i}" for i in range(50)] + [f"xpath_tfidf_{i}" for i in range(25)] + [f"z1_tfidf_{i}" for i in range(20)] + [f"z2_tfidf_{i}" for i in range(20)] + [f"z3_tfidf_{i}" for i in range(20)]

import numpy as np
import pandas as pd
from .multimodal import MultimodalProcessor
from .structural import StructuralProcessor

class HybridProcessor:
    """
    Extracts both Multimodal features (URL and HTML text tokens)
    and Structural features (15-dimensional invariant vector).
    """

    def __init__(self, config: dict):
        self.config = config
        self.multimodal_processor = MultimodalProcessor(config)
        self.structural_processor = StructuralProcessor(config)
        self.fitted = False

    def fit(self, X, y=None):
        self.multimodal_processor.fit(X, y)
        self.structural_processor.fit_transform(X, y)
        self.fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform")

        multi_features = self.multimodal_processor.transform(X)
        struct_features = self.structural_processor.transform(X)
        
        url_features, html_features = multi_features
        
        return [url_features, html_features, struct_features]

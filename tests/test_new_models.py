import pytest
import numpy as np
import pandas as pd
from src.features.manual import ManualFeatureProcessor
from src.features.multimodal import MultimodalProcessor
from src.models.hybrid_svm_knn import SVM_KNN
from src.models.webphish_cnn import WebPhish_CNN

def test_manual_feature_processor():
    config = {"n_features": 48}
    processor = ManualFeatureProcessor(config)
    urls = ["http://example.com", "https://phish-site.net/login"]
    
    features = processor.fit_transform(urls)
    assert features.shape == (2, 48)
    assert isinstance(features, np.ndarray)

def test_svm_knn_fit_predict():
    config = {"svm_C": 1.0, "knn_neighbors": 3}
    model = SVM_KNN(config)
    X = np.random.rand(10, 48)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)
    
    probs = model.predict_proba(X)
    assert probs.shape == (10, 2)

def test_multimodal_processor():
    config = {
        "url_max_length": 20, 
        "html_max_length": 50,
        "html_vocab_size": 100
    }
    processor = MultimodalProcessor(config)
    data = pd.DataFrame({
        "url": ["http://a.com", "http://b.com"],
        "html": ["<html>1</html>", "<html>2</html>"]
    })
    
    features = processor.fit_transform(data)
    assert len(features) == 2
    assert features[0].shape == (2, 20)
    assert features[1].shape == (2, 50)
    assert processor.html_actual_vocab_size > 0

def test_webphish_cnn_fit_predict():
    config = {
        "url_max_length": 20,
        "html_max_length": 50,
        "url_vocab_size": 130,
        "html_vocab_size": 100,
        "filters": 32,
        "kernel_size": 8,
        "batch_size": 2,
        "epochs": 1
    }
    model = WebPhish_CNN(config)
    X_url = np.random.randint(0, 130, (4, 20))
    X_html = np.random.randint(0, 100, (4, 50))
    X = [X_url, X_html]
    y = np.array([0, 1, 0, 1])
    
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (4,)
    
    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)

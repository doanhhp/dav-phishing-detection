import joblib
import os

models = [
    "webphish_cnn", "egso_cnn", "lstm_url", "rnn_gru", "hybrid_svm_knn", 
    "url_rf", "structural_rf", "structural_xgb", "structural_stacking", "mid_fusion_xgb"
]

for m in models:
    path = f"data/processed/{m}/processor.joblib"
    if os.path.exists(path):
        processor = joblib.load(path)
        if hasattr(processor, 'scaler'):
            print(f"{m}: scaler expects {processor.scaler.n_features_in_} features")
        elif hasattr(processor, 'vectorizer'):
            print(f"{m}: vectorizer")
        else:
            print(f"{m}: no scaler")
    else:
        print(f"{m}: no processor found")

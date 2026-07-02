import sys
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def evaluate_cv(model, X, y, name):
    print(f"Running 5-fold CV for {name}...")
    y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
    acc = accuracy_score(y, y_pred) * 100
    prec = precision_score(y, y_pred) * 100
    rec = recall_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred) * 100
    print(f"{name:20s} | {acc:.2f}% | {prec:.2f}% | {rec:.2f}% | {f1:.2f}%")

def main():
    print("Loading Dataset-2021 (X_train)...")
    X = joblib.load('data/processed/mid_fusion_xgb/X_train.joblib')
    y = joblib.load('data/processed/mid_fusion_xgb/y_train.joblib')
    
    print("Loading MidFusion architecture...")
    model_wrapper = joblib.load('experiments/mid_fusion_xgb/model.joblib')
    stacking_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    
    url_pipeline = stacking_model.estimators[0][1]
    html_pipeline = stacking_model.estimators[1][1]
    
    print(f"\n{'Model Architecture':25s} | Accuracy | Precision | Recall | F1-Score")
    print("-" * 75)
    
    evaluate_cv(url_pipeline, X, y, "Level-0 URL Expert")
    evaluate_cv(html_pipeline, X, y, "Level-0 HTML Expert")
    
    # Evaluate Final Mid-Fusion
    evaluate_cv(stacking_model, X, y, "Level-1 Mid-Fusion")

if __name__ == '__main__':
    main()

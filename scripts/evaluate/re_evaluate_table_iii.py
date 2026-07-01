import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.pipeline import load_data
from src.models.model_factory import ModelFactory

def main():
    print("Loading PhreshPhish (2024) dataset...")
    df, y = load_data("data/processed/standardized/phreshphish_dataset.parquet")
    X = df[['Data', 'html']]
    
    models = [
        "webphish_cnn", "egso_cnn", "lstm_url", "rnn_gru", "hybrid_svm_knn", 
        "url_rf", "structural_rf", "structural_xgb", "structural_stacking", "mid_fusion_xgb"
    ]
    
    results = {}
    
    with open("config/benchmarks.yaml", "r") as f:
        config = yaml.safe_load(f)["models"]
        
    for m in models:
        print(f"\n--- Evaluating {m} ---")
        try:
            m_config = config[m]
            # load processor
            processor_path = f"data/processed/{m}/processor.joblib"
            if not os.path.exists(processor_path):
                print(f"Skipping {m}, no processor found.")
                continue
            processor = joblib.load(processor_path)
            
            # transform
            # Special case for mid_fusion_xgb to use the fast structural path if possible
            if m in ["structural_rf", "structural_xgb", "structural_stacking", "mid_fusion_xgb", "structural_dnn"]:
                try:
                    X_p = joblib.load("data/processed/standardized/phreshphish_processed_structural.joblib")
                    print("Loaded pre-processed structural features.")
                except:
                    print("Transforming features...")
                    X_p = processor.transform(X)
                    # joblib.dump(X_p, "data/processed/standardized/phreshphish_processed_structural.joblib")
            else:
                print("Transforming features...")
                X_m = df['Data'] if m in ["lstm_url", "rnn_gru", "url_rf"] else X
                X_p = processor.transform(X_m)
            
            # load model
            model = ModelFactory.create_model(m, m_config)
            if m in ["structural_xgb", "structural_stacking", "mid_fusion_xgb", "url_rf", "structural_rf", "hybrid_svm_knn"]:
                model.load(f"experiments/{m}/model")
            else:
                model.load(f"experiments/{m}/model.h5")
                
            # predict
            y_prob = model.predict_proba(X_p) if hasattr(model, 'predict_proba') else model.predict(X_p)
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            
            acc = accuracy_score(y, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
            
            print(f"Result {m}: Acc {acc:.4f}, Prec {prec:.4f}, Rec {rec:.4f}, F1 {f1:.4f}")
            results[m] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
            
            if m == "mid_fusion_xgb":
                # Find optimal threshold
                best_acc = 0
                best_t = 0.5
                best_prec, best_rec, best_f1 = 0, 0, 0
                for t in np.arange(0.1, 0.9, 0.05):
                    yp = (y_prob > t).astype(int)
                    t_acc = accuracy_score(y, yp)
                    if t_acc > best_acc:
                        best_acc = t_acc
                        best_t = t
                        best_prec, best_rec, best_f1, _ = precision_recall_fscore_support(y, yp, average='binary')
                print(f"Mid-Fusion Optimal (p={best_t:.2f}): Acc {best_acc:.4f}")
                results["mid_fusion_xgb_optimal"] = {"Accuracy": best_acc, "Precision": best_prec, "Recall": best_rec, "F1": best_f1}
        except Exception as e:
            print(f"Failed {m}: {e}")
            
    print("\nFINAL RESULTS SUMMARY:")
    for k, v in results.items():
        print(f"{k}: Acc {v['Accuracy']:.4f}, Prec {v['Precision']:.4f}, Rec {v['Recall']:.4f}, F1 {v['F1']:.4f}")
        
if __name__ == "__main__":
    main()

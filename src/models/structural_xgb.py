import joblib
import xgboost as xgb
from src.models.base import BaseModel

class Structural_XGB(BaseModel):
    """
    XGBoost model that uses the 15-dimensional structural feature vector.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = self._build_model()
        
    def _build_model(self):
        return xgb.XGBClassifier(
            n_estimators=self.config.get("n_estimators", 200),
            learning_rate=self.config.get("learning_rate", 0.1),
            max_depth=self.config.get("max_depth", 5),
            random_state=self.config.get("random_state", 42),
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def save(self, path: str):
        joblib.dump(self.model, f"{path}.joblib")
        
    def load(self, path: str):
        self.model = joblib.load(f"{path}.joblib")
        self.trained = True

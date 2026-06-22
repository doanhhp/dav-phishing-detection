"""Deep Feature-Weighted Stacking (Mid-Fusion) Ensemble."""

from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)

class MidFusionXGB(BaseModel):
    """Stacking model that simulates Mid-Fusion by feeding original features + expert predictions into a final XGBoost meta-model."""

    def __init__(self, config: dict):
        super().__init__(config)
        
        url_params = config.get("url_xgb", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})
        html_params = config.get("html_xgb", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})
        joint_params = config.get("joint_xgb", {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05})
        
        self.url_expert = XGBClassifier(
            **url_params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        
        self.html_expert = XGBClassifier(
            **html_params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        
        self.joint_expert = XGBClassifier(
            **joint_params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        
        self.model = None

    def fit(self, X, y):
        """Train the Mid-Fusion model."""
        num_cols = X.shape[1]
        
        # URL Expert sees only first 10 columns
        url_preprocessor = ColumnTransformer(
            transformers=[('url', 'passthrough', slice(0, 10))]
        )
        url_pipeline = Pipeline(steps=[('prep', url_preprocessor), ('model', self.url_expert)])
        
        # HTML Expert sees columns 10 to N
        html_preprocessor = ColumnTransformer(
            transformers=[('html', 'passthrough', slice(10, num_cols))]
        )
        html_pipeline = Pipeline(steps=[('prep', html_preprocessor), ('model', self.html_expert)])
        
        estimators = [
            ('url_expert', url_pipeline),
            ('html_expert', html_pipeline)
        ]
        
        # passthrough=True concatenates the raw input X alongside the predictions of the experts!
        # This allows the final_estimator (joint_expert) to learn synergistic interaction rules 
        # between raw features and high-level expert predictions (simulating Mid-Fusion).
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.joint_expert,
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
        
        logger.info(f"Training MidFusionXGB with Passthrough Feature-Weighted Stacking...")
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def save(self, path: str):
        if not self.trained:
            logger.warning("Saving an untrained model")
        joblib.dump(self.model, f"{path}.joblib")

    def load(self, path: str):
        try:
            self.model = joblib.load(f"{path}.joblib")
            self.trained = True
        except Exception as e:
            logger.error(f"Failed to load MidFusionXGB model: {e}")
            raise e

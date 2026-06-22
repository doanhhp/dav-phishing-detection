"""Late Fusion Stacking Ensemble."""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)

class StructuralStacking(BaseModel):
    """Stacking model splitting URL and HTML features into separate XGBoost experts."""

    def __init__(self, config: dict):
        super().__init__(config)
        
        url_params = config.get("url_xgb", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})
        html_params = config.get("html_xgb", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})
        
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
        
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.model = None

    def fit(self, X, y):
        """Train the Stacking model."""
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
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        logger.info(f"Training StackingClassifier with URL and HTML Experts...")
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
            logger.error(f"Failed to load StructuralStacking model: {e}")
            raise e

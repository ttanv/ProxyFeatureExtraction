import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb

class XGBoostClassifier:
    """A wrapper for the XGBoost classifier."""
    def __init__(self, params: dict):
        """
        Initializes the XGBoost classifier with given parameters.

        Args:
            params (dict): A dictionary of hyperparameters for the XGBoost model.
        """
        self.model = xgb.XGBClassifier(**params)
        
    @property
    def feature_importances_(self):
        """Exposes the feature importances from the underlying XGBoost model."""
        return self.model.feature_importances_

    def fit(self, *args, **kwargs):
        """Trains the model."""
        self.model.fit(*args, **kwargs)
        return self

    def predict(self, X_test):
        """Makes predictions."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Makes probability predictions."""
        return self.model.predict_proba(X_test)

    def save(self, filepath: str):
        """Saves the model to a file."""
        self.model.save_model(filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads a model from a file."""
        model = xgb.XGBClassifier()
        model.load_model(filepath)
        # We need to wrap it back into our class structure
        xgb_wrapper = cls(params={})
        xgb_wrapper.model = model
        return xgb_wrapper

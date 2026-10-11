"""
Machine Learning Handover-Risk Predictor for TransitConnect.
Trains, evaluates, and deploys Random Forest and Gradient Boosting models
to predict impending handover risk from the normalized 6-feature vector.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split

from .weighted_sum import RiskState
from .dataset_gen import FEATURE_COLUMNS
from ..monitoring.features import NormalizedFeatures, clamp

LABEL_TO_STATE = {
    0: RiskState.LOW,
    1: RiskState.MODERATE,
    2: RiskState.HIGH
}

class MLRiskPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = FEATURE_COLUMNS
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train_models(self, dataset_df: pd.DataFrame, model_type: str = "rf") -> Dict[str, Any]:
        """
        Trains and evaluates ML models using train_test_split.
        model_type: 'rf' (Random Forest) or 'gbdt' (Gradient Boosting)
        """
        X = dataset_df[self.feature_names].values
        y_class = dataset_df["risk_label"].values
        y_score = dataset_df["risk_score_target"].values

        X_train, X_test, y_c_train, y_c_test, y_s_train, y_s_test = train_test_split(
            X, y_class, y_score, test_size=0.25, random_state=42, stratify=y_class
        )

        if model_type == "gbdt":
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

        clf.fit(X_train, y_c_train)
        preds_c = clf.predict(X_test)
        acc = float(accuracy_score(y_c_test, preds_c))
        f1 = float(f1_score(y_c_test, preds_c, average="macro"))

        # Regressor for smooth continuous risk score
        reg = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42)
        reg.fit(X_train, y_s_train)
        preds_s = reg.predict(X_test)
        mse = float(mean_squared_error(y_s_test, preds_s))

        importances = dict(zip(self.feature_names, [float(x) for x in clf.feature_importances_]))

        self.model = {
            "classifier": clf,
            "regressor": reg,
            "model_type": model_type,
            "accuracy": acc,
            "macro_f1": f1,
            "mse": mse,
            "feature_importances": importances
        }

        return {
            "model_type": model_type,
            "accuracy": acc,
            "macro_f1": f1,
            "mse": mse,
            "feature_importances": importances
        }

    def predict(self, features: NormalizedFeatures) -> Tuple[float, str]:
        """
        Infers risk score H and RiskState using the trained ML model.
        Falls back to weighted heuristic if model not yet trained.
        """
        if self.model is None or "classifier" not in self.model:
            # Fallback heuristic
            score = 0.35 * features.F + 0.25 * features.V + 0.25 * features.P + 0.15 * features.R
            score = clamp(score, 0.0, 1.0)
            state = RiskState.LOW if score < 0.35 else (RiskState.MODERATE if score < 0.70 else RiskState.HIGH)
            return score, state

        x = np.array([[features.F, features.V, features.P, features.R, features.U, features.S]])
        pred_class = int(self.model["classifier"].predict(x)[0])
        pred_score = float(self.model["regressor"].predict(x)[0])
        
        pred_score = clamp(pred_score, 0.0, 1.0)
        state = LABEL_TO_STATE.get(pred_class, RiskState.LOW)

        return pred_score, state

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)

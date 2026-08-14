from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


MODEL_CLASSES = {
    "LinearSVC": LinearSVC,
    "LDA": LinearDiscriminantAnalysis,
    "RF": RandomForestClassifier,
    "DT": DecisionTreeClassifier,
    "XGBoost": XGBClassifier,
}


def make_model(
    model_name: str,
    labels: np.ndarray,
    linear_svc_c: float | None,
    config: dict[str, Any],
) -> tuple[Any, float, dict[str, Any]]:
    parameters = dict(config["models"][model_name])
    threshold = float(parameters.pop("threshold"))
    if model_name != "LDA":
        parameters["random_state"] = int(config["random_seed"])
    if model_name == "LinearSVC":
        if linear_svc_c is None:
            raise ValueError("LinearSVC requires its fixed C parameter")
        parameters["C"] = float(linear_svc_c)
    elif model_name == "XGBoost":
        positive = int(np.sum(labels == 1))
        negative = int(np.sum(labels == 0))
        if positive == 0:
            raise ValueError("XGBoost training fold has no positive samples")
        if parameters.get("scale_pos_weight") == "auto":
            parameters["scale_pos_weight"] = negative / positive
    return MODEL_CLASSES[model_name](**parameters), threshold, parameters


def model_scores(model: Any, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), dtype=float)
    return np.asarray(model.predict_proba(values), dtype=float)[:, 1]

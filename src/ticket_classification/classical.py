"""One unweighted Logistic Regression baseline over existing sparse features."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class LogisticRegressionConfig(BaseModel):
    """Intentionally restrict Phase 2.4 to one reproducible baseline family."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    solver: Literal["lbfgs"] = "lbfgs"
    penalty: Literal["l2"] = "l2"
    C: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_iterations: StrictInt = Field(default=1000, gt=0)
    class_weight: None = None
    random_seed: StrictInt = Field(default=42, ge=0, le=4294967295)


@dataclass(frozen=True)
class TrainingResult:
    model: LogisticRegression
    training_time: float


def validate_features(matrix, labels, feature_count: int) -> None:
    if not issparse(matrix):
        raise ValueError("classifier requires sparse TF-IDF features")
    if matrix.shape != (len(labels), feature_count) or not len(labels):
        raise ValueError("feature dimensions must align with nonempty labels")
    if not np.isfinite(matrix.data).all():
        raise ValueError("feature values must be finite")


def train_logistic_regression(matrix, labels, config=None) -> TrainingResult:
    settings = config or LogisticRegressionConfig()
    validate_features(matrix, labels, matrix.shape[1])
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be nonblank single-class strings")
    if len(set(labels)) < 2:
        raise ValueError("training requires at least two classes")
    # L2 is sklearn's default. Omitting penalty avoids its deprecation in 1.9
    # while retaining compatibility with the repository's sklearn >=1.4 range.
    model = LogisticRegression(
        solver=settings.solver, C=settings.C, max_iter=settings.max_iterations,
        class_weight=settings.class_weight, random_state=settings.random_seed,
    )
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(matrix, labels)
    return TrainingResult(model, perf_counter() - start)


def predict_records(model, matrix, records, class_order):
    """Return traceable predictions and probabilities; time predict only."""
    validate_features(matrix, records, model.n_features_in_)
    if list(model.classes_) != list(class_order):
        raise ValueError("model classes differ from frozen class order")
    start = perf_counter()
    predictions = model.predict(matrix)
    duration = perf_counter() - start
    probabilities = model.predict_proba(matrix)
    if predictions.shape != (len(records),):
        raise ValueError("prediction count differs from records")
    if probabilities.shape != (len(records), len(class_order)):
        raise ValueError("probability dimensions differ from records/classes")
    if (not np.isfinite(probabilities).all() or
        (probabilities < 0).any() or (probabilities > 1).any() or
        not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("invalid probability vectors")
    if not set(predictions).issubset(class_order):
        raise ValueError("unknown predicted labels")
    selected = np.array([list(model.classes_).index(label) for label in predictions])
    if not np.allclose(probabilities[np.arange(len(records)), selected],
                       probabilities.max(axis=1), rtol=0, atol=1e-12):
        raise ValueError("prediction does not match maximum probability")
    rows = [
        {"record_id": record.record_id, "actual_label": record.label,
         "predicted_label": str(prediction), "probability_class_order": list(class_order),
         "probabilities": probability.tolist()}
        for record, prediction, probability in zip(records, predictions, probabilities)
    ]
    timing = {"seconds": duration, "record_count": len(records),
              "seconds_per_record": duration / len(records),
              "includes": "Logistic Regression predict only; excludes TF-IDF transform and predict_proba",
              "iterations": 1, "clock": "perf_counter"}
    return rows, timing


def coefficient_feature_mapping(model, vectorizer):
    """Map every feature to its weight; binary class 0 uses negative weights."""
    names = vectorizer.get_feature_names_out()
    if len(names) != model.n_features_in_ or model.coef_.shape[1] != len(names):
        raise ValueError("coefficient and TF-IDF feature dimensions differ")
    weights = model.coef_
    if len(model.classes_) == 2:
        weights = np.vstack([-weights[0], weights[0]])
    if weights.shape[0] != len(model.classes_):
        raise ValueError("coefficient rows differ from model classes")
    return {str(label): dict(zip(names.tolist(), row.tolist()))
            for label, row in zip(model.classes_, weights)}

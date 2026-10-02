"""Simple unweighted classical baselines over existing sparse features."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class LogisticRegressionConfig(BaseModel):
    """Intentionally restrict Phase 2.4 to one reproducible baseline family."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    solver: Literal["lbfgs"] = "lbfgs"
    penalty: Literal["l2"] = "l2"
    C: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_iterations: StrictInt = Field(default=1000, gt=0)
    class_weight: None = None
    random_seed: StrictInt = Field(default=42, ge=0, le=4294967295)


class LinearSVMConfig(BaseModel):
    """Frozen Phase 2.5 Linear SVM baseline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model_type: Literal["linear_svm"] = "linear_svm"
    C: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    class_weight: None = None
    max_iterations: StrictInt = Field(default=1000, gt=0)
    tolerance: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    dual: Literal["auto"] = "auto"
    random_seed: StrictInt = Field(default=42, ge=0, le=4294967295)


@dataclass(frozen=True)
class TrainingResult:
    model: LogisticRegression
    training_time: float


@dataclass(frozen=True)
class LinearSVMTrainingResult:
    model: LinearSVC
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


def train_linear_svm(matrix, labels, config=None) -> LinearSVMTrainingResult:
    """Fit the frozen unweighted Linear SVM using training features only."""
    settings = config or LinearSVMConfig()
    validate_features(matrix, labels, matrix.shape[1])
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be nonblank single-class strings")
    if len(set(labels)) < 2:
        raise ValueError("training requires at least two classes")
    model = LinearSVC(
        C=settings.C,
        class_weight=settings.class_weight,
        max_iter=settings.max_iterations,
        tol=settings.tolerance,
        dual=settings.dual,
        random_state=settings.random_seed,
    )
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(matrix, labels)
    return LinearSVMTrainingResult(model, perf_counter() - start)


def linear_svm_decision_scores(model, matrix):
    """Return one raw decision margin per model class in model class order."""
    scores = np.asarray(model.decision_function(matrix))
    if scores.ndim == 1:
        # LinearSVC exposes one signed margin for binary classification. Its
        # negative and positive orientations correspond to classes 0 and 1.
        scores = np.column_stack((-scores, scores))
    expected_shape = (matrix.shape[0], len(model.classes_))
    if scores.shape != expected_shape or not np.isfinite(scores).all():
        raise ValueError("decision-score dimensions differ from records/classes")
    return scores


def predict_linear_svm_records(
    model,
    matrix,
    records,
    class_order,
    *,
    warmup_iterations=1,
    measured_iterations=5,
):
    """Return traceable predictions, raw margins, and lightweight CPU timing."""
    validate_features(matrix, records, model.n_features_in_)
    if list(model.classes_) != list(class_order):
        raise ValueError("model classes differ from frozen class order")
    if warmup_iterations < 0 or measured_iterations < 1:
        raise ValueError("timing iterations must include at least one measurement")
    for _ in range(warmup_iterations):
        model.predict(matrix)
    durations = []
    predictions = None
    for _ in range(measured_iterations):
        start = perf_counter()
        predictions = model.predict(matrix)
        durations.append(perf_counter() - start)
    assert predictions is not None
    scores = linear_svm_decision_scores(model, matrix)
    if predictions.shape != (len(records),):
        raise ValueError("prediction count differs from records")
    if not set(predictions).issubset(class_order):
        raise ValueError("unknown predicted labels")
    selected = np.asarray([list(model.classes_).index(label) for label in predictions])
    if not np.allclose(
        scores[np.arange(len(records)), selected],
        scores.max(axis=1),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("prediction does not match maximum decision score")
    rows = [
        {
            "record_id": record.record_id,
            "actual_label": record.label,
            "predicted_label": str(prediction),
            "decision_score_class_order": list(class_order),
            "decision_scores": {
                str(label): float(score)
                for label, score in zip(class_order, score_row)
            },
            "score_type": "raw_linear_svm_decision_function",
            "scores_are_probabilities": False,
        }
        for record, prediction, score_row in zip(records, predictions, scores)
    ]
    aggregate = float(sum(durations))
    timing = {
        "aggregate_seconds": aggregate,
        "mean_iteration_seconds": aggregate / measured_iterations,
        "seconds_per_record": aggregate / (measured_iterations * len(records)),
        "record_count": len(records),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "includes": "Linear SVM predict only; excludes TF-IDF transform and decision_function",
        "clock": "perf_counter",
    }
    return rows, timing


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

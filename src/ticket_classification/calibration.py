"""Model-independent calibration metrics and post-hoc calibrators.

The Brier score convention in this module is the mean, over examples, of the
sum of squared errors over all classes. ECE is top-label ECE with equal-width
bins: bins are [left, right), except the final bin includes confidence 1.0.
Empty bins are retained in reliability data with null summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression


PROBABILITY_TOLERANCE = 1e-6


def validate_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray | None = None,
    *,
    expected_classes: int | None = None,
) -> np.ndarray:
    """Return a validated two-dimensional probability array."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must be a nonempty two-dimensional array")
    if expected_classes is not None and values.shape[1] != expected_classes:
        raise ValueError("probability class dimension differs from the frozen label order")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("probabilities must be within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=PROBABILITY_TOLERANCE):
        raise ValueError("each probability vector must sum to 1")
    if true_label_ids is not None:
        labels = np.asarray(true_label_ids)
        if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
            raise ValueError("true labels must align with probability rows")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("true label IDs must be integers")
        if (labels < 0).any() or (labels >= values.shape[1]).any():
            raise ValueError("true label ID is outside the probability class dimension")
    return values


def _bin_indexes(confidences: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    indexes = np.searchsorted(boundaries, confidences, side="right") - 1
    return np.clip(indexes, 0, bins - 1), boundaries


def build_reliability_bins(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Build deterministic top-label reliability data."""

    values = validate_probabilities(probabilities, true_label_ids)
    labels = np.asarray(true_label_ids, dtype=np.int64)
    predictions = values.argmax(axis=1)
    confidences = values.max(axis=1)
    correct = predictions == labels
    indexes, boundaries = _bin_indexes(confidences, bins)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indexes == index
        count = int(mask.sum())
        mean_confidence = float(confidences[mask].mean()) if count else None
        observed_accuracy = float(correct[mask].mean()) if count else None
        result.append(
            {
                "bin_index": index,
                "lower_bound": float(boundaries[index]),
                "upper_bound": float(boundaries[index + 1]),
                "lower_inclusive": True,
                "upper_inclusive": index == bins - 1,
                "count": count,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "correct_count": int(correct[mask].sum()),
                "incorrect_count": int(count - correct[mask].sum()),
                "confidence_minus_accuracy": (
                    mean_confidence - observed_accuracy if count else None
                ),
            }
        )
    return result


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Calculate equal-width top-label expected calibration error."""

    values = validate_probabilities(probabilities, true_label_ids)
    reliability = build_reliability_bins(values, true_label_ids, bins=bins)
    total = values.shape[0]
    return float(
        sum(
            row["count"] / total
            * abs(row["mean_confidence"] - row["observed_accuracy"])
            for row in reliability
            if row["count"]
        )
    )


def multiclass_brier_score(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
) -> float:
    """Return mean per-example summed squared error across classes."""

    values = validate_probabilities(probabilities, true_label_ids)
    labels = np.asarray(true_label_ids, dtype=np.int64)
    targets = np.zeros_like(values)
    targets[np.arange(values.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((values - targets) ** 2, axis=1)))


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
) -> float:
    """Return multiclass NLL from probabilities with safe clipping."""

    values = validate_probabilities(probabilities, true_label_ids)
    labels = np.asarray(true_label_ids, dtype=np.int64)
    selected = values[np.arange(values.shape[0]), labels]
    return float(-np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean())


def negative_log_likelihood_from_logits(
    logits: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    *,
    temperature: float = 1.0,
) -> float:
    """Return stable multiclass NLL directly from logits."""

    values, labels = _validate_logits(logits, true_label_ids)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values / temperature
    losses = logsumexp(scaled, axis=1) - scaled[np.arange(values.shape[0]), labels]
    return float(losses.mean())


def softmax_logits(
    logits: Sequence[Sequence[float]] | np.ndarray, *, temperature: float = 1.0
) -> np.ndarray:
    values, _ = _validate_logits(logits)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def evaluate_calibration(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Evaluate all probability metrics using one validated input."""

    values = validate_probabilities(probabilities, true_label_ids)
    return {
        "ece": expected_calibration_error(values, true_label_ids, bins=bins),
        "multiclass_brier_score": multiclass_brier_score(values, true_label_ids),
        "negative_log_likelihood": negative_log_likelihood(values, true_label_ids),
        "brier_score_convention": "mean_per_example_sum_over_classes",
        "ece_definition": "equal_width_top_label",
        "confidence_definition": "maximum_class_probability",
        "reliability_bins": build_reliability_bins(values, true_label_ids, bins=bins),
    }


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Positive scalar temperature fitted without changing model weights."""

    temperature: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    def transform(self, logits: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return softmax_logits(logits, temperature=self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "temperature_scaling", "temperature": self.temperature}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemperatureScaler":
        if value.get("method") != "temperature_scaling":
            raise ValueError("artifact is not a temperature-scaling policy")
        return cls(temperature=float(value["temperature"]))


def fit_temperature_scaling(
    logits: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    *,
    minimum_temperature: float = 1e-3,
    maximum_temperature: float = 100.0,
) -> TemperatureScaler:
    """Fit one positive temperature by bounded deterministic NLL minimization."""

    values, labels = _validate_logits(logits, true_label_ids)
    if minimum_temperature <= 0 or maximum_temperature <= minimum_temperature:
        raise ValueError("temperature bounds must be positive and increasing")
    original = values.copy()
    result = minimize_scalar(
        lambda temperature: negative_log_likelihood_from_logits(
            values, labels, temperature=float(temperature)
        ),
        bounds=(minimum_temperature, maximum_temperature),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x) or result.x <= 0:
        raise RuntimeError("temperature optimization failed")
    if not np.array_equal(values, original):
        raise RuntimeError("temperature fitting modified the source logits")
    return TemperatureScaler(float(result.x))


@dataclass(frozen=True, slots=True)
class SigmoidScoreCalibrator:
    """One-vs-rest sigmoid calibrator for frozen multiclass decision scores."""

    class_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercepts: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.class_order)
        if size < 2 or len(set(self.class_order)) != size:
            raise ValueError("class_order must contain at least two unique classes")
        if len(self.coefficients) != size or len(self.intercepts) != size:
            raise ValueError("one sigmoid parameter pair is required per class")
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercepts).all():
            raise ValueError("sigmoid parameters must be finite")

    def transform(self, decision_scores: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        values = np.asarray(decision_scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.class_order):
            raise ValueError("decision-score dimensions differ from the frozen class order")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("decision scores must be nonempty and finite")
        probabilities = expit(
            values * np.asarray(self.coefficients) + np.asarray(self.intercepts)
        )
        totals = probabilities.sum(axis=1, keepdims=True)
        if (totals <= 0).any() or not np.isfinite(totals).all():
            raise ValueError("sigmoid calibration produced invalid probability totals")
        return probabilities / totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "sigmoid_calibration",
            "class_order": list(self.class_order),
            "coefficients": list(self.coefficients),
            "intercepts": list(self.intercepts),
            "multiclass_strategy": "one_vs_rest_sigmoid_then_normalize",
            "argmax_may_change": True,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SigmoidScoreCalibrator":
        if value.get("method") != "sigmoid_calibration":
            raise ValueError("artifact is not a sigmoid-calibration policy")
        return cls(
            class_order=tuple(value["class_order"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercepts=tuple(float(item) for item in value["intercepts"]),
        )


def fit_sigmoid_score_calibration(
    decision_scores: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    class_order: Sequence[str],
    *,
    random_seed: int = 42,
) -> SigmoidScoreCalibrator:
    """Fit only one-vs-rest sigmoid parameters; the base classifier stays frozen."""

    values, labels = _validate_logits(decision_scores, true_label_ids)
    classes = tuple(class_order)
    if values.shape[1] != len(classes):
        raise ValueError("decision-score dimensions differ from class_order")
    coefficients: list[float] = []
    intercepts: list[float] = []
    for class_id in range(len(classes)):
        targets = (labels == class_id).astype(np.int64)
        if np.unique(targets).size != 2:
            raise ValueError("calibration subset must contain positive and negative examples per class")
        model = LogisticRegression(
            solver="lbfgs", C=1e6, max_iter=1000, random_state=random_seed
        )
        model.fit(values[:, class_id].reshape(-1, 1), targets)
        coefficients.append(float(model.coef_[0, 0]))
        intercepts.append(float(model.intercept_[0]))
    return SigmoidScoreCalibrator(classes, tuple(coefficients), tuple(intercepts))


def _validate_logits(
    logits: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits/scores must be a nonempty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("logits/scores must be finite")
    if true_label_ids is None:
        return values, None
    labels = np.asarray(true_label_ids)
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("true labels must align with logits/scores")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("true label IDs must be integers")
    labels = labels.astype(np.int64)
    if (labels < 0).any() or (labels >= values.shape[1]).any():
        raise ValueError("true label ID is outside the class dimension")
    return values, labels

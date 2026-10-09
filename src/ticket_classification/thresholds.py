"""Confidence-based acceptance and rejection for single-label multiclass models."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ticket_classification.calibration import validate_probabilities


DEFAULT_THRESHOLDS = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def apply_threshold(
    probabilities: Sequence[float] | np.ndarray,
    class_order: Sequence[str],
    threshold: float,
    *,
    calibrated: bool,
) -> dict[str, Any]:
    """Apply a policy to one prediction without using its true label."""

    classes = _validate_class_order(class_order)
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 1:
        raise ValueError("one probability vector is required")
    validated = validate_probabilities(value.reshape(1, -1), expected_classes=len(classes))[0]
    _validate_threshold(threshold)
    predicted_id = int(validated.argmax())
    confidence = float(validated[predicted_id])
    return {
        "predicted_label": classes[predicted_id],
        "predicted_label_id": predicted_id,
        "confidence": confidence,
        "decision": "accepted" if confidence >= threshold else "needs_review",
        "threshold": float(threshold),
        "calibrated": bool(calibrated),
    }


def evaluate_threshold(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    class_order: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one top-label confidence threshold."""

    classes = _validate_class_order(class_order)
    values = validate_probabilities(
        probabilities, true_label_ids, expected_classes=len(classes)
    )
    labels = np.asarray(true_label_ids, dtype=np.int64)
    _validate_threshold(threshold)
    predicted = values.argmax(axis=1)
    confidence = values.max(axis=1)
    accepted = confidence >= threshold
    correct = predicted == labels
    total = values.shape[0]
    accepted_count = int(accepted.sum())
    rejected_count = total - accepted_count
    correct_accepted = int((accepted & correct).sum())
    incorrect_accepted = accepted_count - correct_accepted
    selective_accuracy = (
        correct_accepted / accepted_count if accepted_count else None
    )
    per_class: dict[str, Any] = {}
    for class_id, label in enumerate(classes):
        actual_mask = labels == class_id
        class_total = int(actual_mask.sum())
        class_accepted = actual_mask & accepted
        accepted_for_class = int(class_accepted.sum())
        per_class[label] = {
            "total_count": class_total,
            "accepted_count": accepted_for_class,
            "rejected_count": class_total - accepted_for_class,
            "coverage": accepted_for_class / class_total if class_total else None,
            "accepted_error_count": int((class_accepted & ~correct).sum()),
        }
    return {
        "threshold": float(threshold),
        "confidence_definition": "maximum_class_probability",
        "acceptance_rule": "confidence >= threshold",
        "total_count": total,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "correct_accepted_count": correct_accepted,
        "incorrect_accepted_count": incorrect_accepted,
        "coverage": accepted_count / total,
        "rejection_rate": rejected_count / total,
        "selective_accuracy": selective_accuracy,
        "selective_error_rate": (
            incorrect_accepted / accepted_count if accepted_count else None
        ),
        "per_class": per_class,
    }


def evaluate_thresholds(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    true_label_ids: Sequence[int] | np.ndarray,
    class_order: Sequence[str],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Evaluate a deterministic ordered threshold grid, including accept-all."""

    if not thresholds:
        raise ValueError("at least one threshold is required")
    normalized = [float(value) for value in thresholds]
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must be unique")
    if normalized != sorted(normalized):
        raise ValueError("thresholds must be in increasing order")
    for value in normalized:
        _validate_threshold(value)
    if 0.0 not in normalized:
        normalized.insert(0, 0.0)
    return [
        evaluate_threshold(probabilities, true_label_ids, class_order, threshold)
        for threshold in normalized
    ]


def select_threshold_for_accuracy(
    evaluations: Sequence[dict[str, Any]],
    minimum_selective_accuracy: float,
) -> dict[str, Any] | None:
    """Maximize coverage subject to a caller-supplied accuracy requirement."""

    if (
        not np.isfinite(minimum_selective_accuracy)
        or minimum_selective_accuracy < 0
        or minimum_selective_accuracy > 1
    ):
        raise ValueError("minimum selective accuracy must be within [0, 1]")
    eligible = [
        row
        for row in evaluations
        if row.get("selective_accuracy") is not None
        and row["selective_accuracy"] >= minimum_selective_accuracy
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (row["coverage"], row["selective_accuracy"], -row["threshold"]),
    )


def _validate_class_order(class_order: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(class_order)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain at least two unique labels")
    if any(not isinstance(label, str) or not label.strip() for label in classes):
        raise ValueError("class labels must be nonblank strings")
    return classes


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("threshold must be finite and within [0, 1]")

import pytest

from ticket_classification.thresholds import (
    apply_threshold,
    evaluate_threshold,
    evaluate_thresholds,
    select_threshold_for_accuracy,
)


CLASSES = ("Change", "Incident", "Problem", "Request")


def test_threshold_boundary_is_accepted_and_label_is_preserved():
    result = apply_threshold(
        [0.1, 0.6, 0.2, 0.1], CLASSES, 0.6, calibrated=True
    )
    assert result == {
        "predicted_label": "Incident",
        "predicted_label_id": 1,
        "confidence": 0.6,
        "decision": "accepted",
        "threshold": 0.6,
        "calibrated": True,
    }


def test_rejected_prediction_remains_a_model_class():
    result = apply_threshold(
        [0.1, 0.4, 0.3, 0.2], CLASSES, 0.5, calibrated=False
    )
    assert result["predicted_label"] == "Incident"
    assert result["decision"] == "needs_review"


def test_coverage_selective_accuracy_and_per_class_counts():
    probabilities = [
        [0.90, 0.04, 0.03, 0.03],
        [0.20, 0.60, 0.10, 0.10],
        [0.10, 0.70, 0.15, 0.05],
        [0.10, 0.20, 0.30, 0.40],
    ]
    result = evaluate_threshold(probabilities, [0, 1, 2, 3], CLASSES, 0.65)
    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 2
    assert result["coverage"] == pytest.approx(0.5)
    assert result["selective_accuracy"] == pytest.approx(0.5)
    assert result["selective_error_rate"] == pytest.approx(0.5)
    assert result["per_class"]["Problem"]["accepted_error_count"] == 1
    assert result["per_class"]["Request"]["rejected_count"] == 1


def test_zero_accepted_predictions_are_undefined_not_perfect():
    result = evaluate_threshold(
        [[0.4, 0.3, 0.2, 0.1]], [0], CLASSES, 0.95
    )
    assert result["accepted_count"] == 0
    assert result["selective_accuracy"] is None
    assert result["selective_error_rate"] is None


def test_threshold_sweep_adds_accept_all_and_is_deterministic():
    args = ([[0.7, 0.1, 0.1, 0.1]], [0], CLASSES)
    first = evaluate_thresholds(*args, thresholds=[0.5, 0.8])
    second = evaluate_thresholds(*args, thresholds=[0.5, 0.8])
    assert first == second
    assert first[0]["threshold"] == 0.0
    assert first[0]["coverage"] == 1.0


def test_policy_selection_maximizes_coverage_under_accuracy_constraint():
    evaluations = [
        {"threshold": 0.5, "coverage": 0.9, "selective_accuracy": 0.90},
        {"threshold": 0.7, "coverage": 0.7, "selective_accuracy": 0.96},
        {"threshold": 0.8, "coverage": 0.5, "selective_accuracy": 0.98},
    ]
    selected = select_threshold_for_accuracy(evaluations, 0.95)
    assert selected["threshold"] == 0.7
    assert select_threshold_for_accuracy(evaluations, 0.99) is None


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        apply_threshold([0.7, 0.1, 0.1, 0.1], CLASSES, threshold, calibrated=False)

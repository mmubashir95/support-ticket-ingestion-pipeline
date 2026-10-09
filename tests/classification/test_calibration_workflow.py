import json

import numpy as np
import pytest

from ticket_classification.calibration import fit_temperature_scaling
from ticket_classification.calibration_workflow import (
    _calibration_decision,
    _calibration_scores,
    _require_argmax_preserved,
    apply_prediction_decision,
)


RNG = np.random.default_rng(7)
SCORES = RNG.normal(size=(200, 4)) * 2.0
LABELS = np.where(RNG.random(200) < 0.7, SCORES.argmax(axis=1), RNG.integers(0, 4, 200))


@pytest.mark.parametrize("model", ["transformer", "logistic_regression", "linear_svm"])
def test_selected_calibration_never_changes_predicted_class(model):
    if model == "logistic_regression":
        exp = np.exp(SCORES - SCORES.max(axis=1, keepdims=True))
        subset = {"probabilities": exp / exp.sum(axis=1, keepdims=True), "logits_or_scores": None}
    else:
        subset = {"probabilities": None, "logits_or_scores": SCORES}
    scores = _calibration_scores(model, subset)
    scaler = fit_temperature_scaling(scores, LABELS)
    calibrated = scaler.transform(scores)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), SCORES.argmax(axis=1))
    _require_argmax_preserved(scores, calibrated)


def test_argmax_guard_rejects_changed_prediction():
    with pytest.raises(RuntimeError, match="changed a predicted class"):
        _require_argmax_preserved(np.array([[2.0, 1.0]]), np.array([[0.4, 0.6]]))


def test_calibration_decision_rejects_changed_classification_metrics():
    metrics = {"ece": 0.1, "multiclass_brier_score": 0.2, "negative_log_likelihood": 0.3}
    with pytest.raises(RuntimeError, match="macro_f1"):
        _calibration_decision(
            metrics, metrics,
            {"accuracy": 0.8, "macro_f1": 0.8},
            {"accuracy": 0.8, "macro_f1": 0.79},
        )


def test_logistic_regression_decision_preserves_base_prediction(tmp_path):
    parameter = tmp_path / "lr_temperature.json"
    parameter.write_text(
        json.dumps({"method": "temperature_scaling", "temperature": 0.5}),
        encoding="utf-8",
    )
    result = apply_prediction_decision(
        "logistic_regression",
        [0.1, 0.2, 0.45, 0.25],
        {"models": {"logistic_regression": {
            "calibration_method": "temperature_scaling",
            "parameter_reference": str(parameter),
        }}},
        {"models": {"logistic_regression": {"status": "selected", "selected_threshold": 0.5}}},
    )
    assert result["predicted_label"] == "Problem"
    assert result["confidence"] > 0.45


def test_decision_interface_does_not_require_ground_truth(tmp_path):
    parameter = tmp_path / "temperature.json"
    parameter.write_text(
        json.dumps({"method": "temperature_scaling", "temperature": 2.0}),
        encoding="utf-8",
    )
    calibration_policy = {
        "models": {
            "transformer": {
                "calibration_method": "temperature_scaling",
                "parameter_reference": str(parameter),
            }
        }
    }
    threshold_policy = {
        "models": {
            "transformer": {
                "status": "selected",
                "selected_threshold": 0.6,
            }
        }
    }
    result = apply_prediction_decision(
        "transformer",
        [0.0, 3.0, 1.0, -1.0],
        calibration_policy,
        threshold_policy,
    )
    assert result["predicted_label"] == "Incident"
    assert result["calibrated"] is True
    assert result["decision"] in {"accepted", "needs_review"}
    assert "true_label" not in result


def test_decision_interface_refuses_pending_business_threshold():
    calibration_policy = {
        "models": {
            "logistic_regression": {
                "calibration_method": "none",
                "parameter_reference": None,
            }
        }
    }
    threshold_policy = {
        "models": {
            "logistic_regression": {
                "status": "pending",
                "selected_threshold": None,
            }
        }
    }
    with pytest.raises(ValueError, match="pending"):
        apply_prediction_decision(
            "logistic_regression",
            [0.1, 0.7, 0.1, 0.1],
            calibration_policy,
            threshold_policy,
        )

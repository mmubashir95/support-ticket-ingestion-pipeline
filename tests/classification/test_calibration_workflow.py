import json

import pytest

from ticket_classification.calibration_workflow import apply_prediction_decision


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

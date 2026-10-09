"""Phase 2.9 deterministic weighting and pending-comparison tests."""

import json

import pytest

from ticket_classification.class_weighting import (
    _metric_comparison,
    calculate_balanced_class_weights,
    run_weighting_comparison,
)
from ticket_classification.classical import LinearSVMConfig, LogisticRegressionConfig


LABELS = ("Change", "Incident", "Problem", "Request")


def test_balanced_weights_use_frozen_order_and_formula() -> None:
    labels = ["Change", "Incident", "Incident", "Problem", "Request", "Request", "Request", "Request"]
    weights = calculate_balanced_class_weights(labels, LABELS)
    assert list(weights) == list(LABELS)
    assert weights == {
        "Change": 2.0,
        "Incident": 1.0,
        "Problem": 2.0,
        "Request": 0.5,
    }


@pytest.mark.parametrize(
    "labels,class_order,message",
    [
        ([], LABELS, "must not be empty"),
        (["Change", "Unknown"], LABELS, "unknown classes"),
        (["Change", "Incident"], LABELS, "missing frozen classes"),
        (["Change", "Incident"], ("Change", "Change"), "unique labels"),
    ],
)
def test_balanced_weights_reject_invalid_inputs(labels, class_order, message) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_balanced_class_weights(labels, class_order)


def test_classical_configs_change_only_class_weight() -> None:
    unweighted_lr = LogisticRegressionConfig()
    weighted_lr = LogisticRegressionConfig(class_weight="balanced")
    unweighted_svm = LinearSVMConfig()
    weighted_svm = LinearSVMConfig(class_weight="balanced")
    assert unweighted_lr.model_copy(update={"class_weight": "balanced"}) == weighted_lr
    assert unweighted_svm.model_copy(update={"class_weight": "balanced"}) == weighted_svm


def _metrics(macro_f1: float) -> dict:
    return {
        "class_order": list(LABELS),
        "accuracy": 0.7,
        "macro_precision": 0.6,
        "macro_recall": 0.5,
        "macro_f1": macro_f1,
        "weighted_f1": 0.65,
        "per_class": {
            label: {"precision": 0.6, "recall": 0.5, "f1": macro_f1, "support": 1}
            for label in LABELS
        },
    }


def test_metric_comparison_calculates_summary_and_per_class_deltas() -> None:
    result = _metric_comparison(_metrics(0.4), _metrics(0.55))
    assert result["summary"]["macro_f1"]["difference"] == pytest.approx(0.15)
    assert result["per_class"]["Problem"]["f1"]["difference"] == pytest.approx(0.15)


def test_comparison_keeps_missing_weighted_experiments_pending(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    weighting = tmp_path / "weighting"
    for name in ("logistic_regression", "linear_svm"):
        directory = baseline / name
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps({"validation": _metrics(0.4)}), encoding="utf-8"
        )
    transformer = baseline / "transformer_training"
    transformer.mkdir(parents=True)
    (transformer / "validation_metrics.json").write_text(
        json.dumps(_metrics(0.5)), encoding="utf-8"
    )
    result = run_weighting_comparison(baseline, weighting)
    assert result["phase_complete"] is False
    assert all(value["status"] == "PENDING" for value in result["final_weighting_policy"].values())
    persisted = json.loads((weighting / "final_weighting_policy.json").read_text())
    assert persisted["distilbert"]["selected_weighting_policy"] is None

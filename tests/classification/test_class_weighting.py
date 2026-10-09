"""Phase 2.9 deterministic weighting and pending-comparison tests."""

import json
from pathlib import Path

import pytest

from ticket_classification.class_weighting import (
    _metric_comparison,
    _validate_transformer_experiments,
    _validation_metrics,
    calculate_balanced_class_weights,
    run_weighting_comparison,
)
from ticket_classification.classical import LinearSVMConfig, LogisticRegressionConfig


LABELS = ("Change", "Incident", "Problem", "Request")
RUN_ROOT = Path("artifacts/classification/run")


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


def test_validation_selection_does_not_read_test_metrics(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"validation": _metrics(0.4), "test": _metrics(1.0)}),
        encoding="utf-8",
    )
    assert _validation_metrics(path, nested=True)["macro_f1"] == 0.4


def test_persisted_transformer_experiments_are_controlled_and_match_expected_deltas() -> None:
    evidence = _validate_transformer_experiments(RUN_ROOT, RUN_ROOT / "class_weighting")
    assert evidence["dataset_version"] == "ds_6fa2f19cf683"
    assert evidence["split_version"] == "split_d1f67d41500f"
    assert evidence["transformer_dataset_version"] == "transformer_ds_340aff24d262"
    assert evidence["label_order"] == list(LABELS)
    unweighted = json.loads((RUN_ROOT / "transformer_training/validation_metrics.json").read_text())
    weighted = json.loads((RUN_ROOT / "class_weighting/transformer/validation_metrics.json").read_text())
    comparison = _metric_comparison(unweighted, weighted)
    assert comparison["summary"]["accuracy"]["difference"] == pytest.approx(-0.013992537313432863)
    assert comparison["summary"]["macro_precision"]["difference"] == pytest.approx(-0.0117449260559813)
    assert comparison["summary"]["macro_recall"]["difference"] == pytest.approx(0.0069229717826442)
    assert comparison["summary"]["macro_f1"]["difference"] == pytest.approx(-0.002729833618667632)
    assert comparison["summary"]["weighted_f1"]["difference"] == pytest.approx(-0.00834854721040101)


def test_final_policy_selects_unweighted_transformer_and_balanced_classical_models() -> None:
    policy = json.loads(
        (RUN_ROOT / "class_weighting/final_weighting_policy.json").read_text(encoding="utf-8")
    )
    assert policy["distilbert"]["selected_weighting_policy"] == "none"
    assert policy["distilbert"]["selected_experiment"] == "unweighted"
    assert policy["distilbert"]["selected_checkpoint_reference"].endswith(
        "transformer_training/checkpoints/checkpoint-3753"
    )
    assert policy["distilbert"]["final_test_evaluation_status"] == "PENDING"
    assert policy["logistic_regression"]["selected_weighting_policy"] == "balanced"
    assert policy["linear_svm"]["selected_weighting_policy"] == "balanced"


def test_selected_unweighted_checkpoint_loads_with_frozen_labels() -> None:
    transformers = pytest.importorskip("transformers")
    checkpoint = RUN_ROOT / "transformer_training/checkpoints/checkpoint-3753"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    assert model.config.num_labels == 4
    assert model.config.label2id == {label: index for index, label in enumerate(LABELS)}
    assert model.config.id2label == {index: label for index, label in enumerate(LABELS)}
    assert tokenizer.name_or_path == str(checkpoint)

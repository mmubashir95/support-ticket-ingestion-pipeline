import pytest

from ticket_classification.final_comparison import (
    _benchmark_summary, align_predictions, paired_bootstrap_macro_f1,
    paired_significance, render_final_report,
)


def test_prediction_alignment_uses_record_identity_not_row_order():
    rows = [
        {"record_id": "b", "true_label": "Y", "predicted_label": "X"},
        {"record_id": "a", "true_label": "X", "predicted_label": "X"},
    ]
    assert align_predictions(["a", "b"], ["X", "Y"], rows, "model") == ["X", "X"]


def test_prediction_alignment_rejects_membership_and_label_drift():
    with pytest.raises(ValueError, match="membership"):
        align_predictions(["a"], ["X"], [{"record_id": "b", "true_label": "X", "predicted_label": "X"}], "model")
    with pytest.raises(ValueError, match="true label"):
        align_predictions(["a"], ["X"], [{"record_id": "a", "true_label": "Y", "predicted_label": "X"}], "model")


def test_paired_evidence_is_deterministic():
    actual = ["A", "A", "B", "B"]
    left = ["A", "A", "B", "B"]
    right = ["A", "B", "A", "B"]
    evidence = paired_significance(actual, left, right)
    assert evidence["left_only_correct"] == 2
    assert evidence["right_only_correct"] == 0
    first = paired_bootstrap_macro_f1(actual, left, right, ["A", "B"], repetitions=20, seed=7)
    assert first == paired_bootstrap_macro_f1(actual, left, right, ["A", "B"], repetitions=20, seed=7)
    assert first["ci95_lower"] >= 0


def test_benchmark_summary_does_not_invent_missing_cpu_results():
    data = {"sample": {"split": "test"}, "results": [{
        "model": "transformer", "device": "mps", "artifact_size": {"total_bytes": 10},
        "measurements": [{"batch_size": 1, "p50_batch_seconds": 0.01,
                          "p95_batch_seconds": 0.02, "cost": None}],
    }]}
    summary = _benchmark_summary(data)
    assert summary["transformer"]["gpu_p50_ms"] == 10
    assert summary["transformer"]["cpu_p50_ms"] is None
    assert summary["linear_svm"]["model_size_bytes"] is None


def test_report_marks_partial_measurements_as_na():
    model = {"quality": {"macro_f1": .8, "macro_precision": .8, "macro_recall": .8,
                         "weighted_f1": .8, "per_class": {"A": {"f1": .8}}},
             "training_time_seconds": 1.0,
             "engineering": {"cpu_p50_ms": None, "cpu_p95_ms": None, "gpu_p50_ms": None,
                             "model_size_bytes": 10, "cost_per_1000": None},
             "explainability": "High", "probability_output": "Native",
             "calibration": {"validation_threshold_selection_ece": .1},
             "operational_complexity": "Low"}
    data = {"models": {name: model for name in ("logistic_regression", "linear_svm", "transformer")},
            "recommendation": {"display_name": "TF-IDF + Linear SVM", "rationale": "test"},
            "evaluation_split": "test", "test_record_count": 4, "label_order": ["A"],
            "pairwise_evidence": {}, "missing_benchmark_evidence": ["CPU"],
            "threshold_status": "pending", "limitations": ["limit"]}
    report = render_final_report(data)
    assert "Selected model: TF-IDF + Linear SVM" in report
    assert "N/A" in report
    assert "Phase 2 freeze" in report

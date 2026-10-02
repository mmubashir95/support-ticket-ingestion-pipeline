"""Focused Phase 2.5 Linear SVM baseline tests."""

import json

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer

from test_classical import persist_inputs
from test_features import frozen_splits
from ticket_classification.classical import (
    LinearSVMConfig,
    linear_svm_decision_scores,
    predict_linear_svm_records,
    train_linear_svm,
)
from ticket_classification.classical_outputs import (
    load_linear_svm,
    run_linear_svm_baseline,
)
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.features import TfidfConfig, build_tfidf_features


@pytest.mark.parametrize(
    "values",
    [
        {"model_type": "svm"},
        {"C": 0},
        {"C": -1},
        {"C": float("inf")},
        {"C": float("nan")},
        {"class_weight": "balanced"},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"tolerance": 0},
        {"dual": True},
        {"random_seed": -1},
        {"unknown": 1},
    ],
)
def test_invalid_linear_svm_configuration(values):
    with pytest.raises(ValidationError):
        LinearSVMConfig(**values)


def test_linear_svm_configuration_roundtrip():
    config = LinearSVMConfig()
    assert config.model_dump() == {
        "model_type": "linear_svm",
        "C": 1.0,
        "class_weight": None,
        "max_iterations": 1000,
        "tolerance": 1e-4,
        "dual": "auto",
        "random_seed": 42,
    }
    assert LinearSVMConfig.model_validate_json(config.model_dump_json()) == config


def test_linear_svm_trains_predicts_scores_and_is_reproducible():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [
            "apple red fruit",
            "apple sweet fruit",
            "banana yellow fruit",
            "banana ripe fruit",
            "cherry red stone",
            "cherry tart stone",
        ]
    )
    labels = ["A", "A", "B", "B", "C", "C"]
    first = train_linear_svm(matrix, labels)
    second = train_linear_svm(matrix, labels)
    assert first.training_time >= 0
    assert list(first.model.classes_) == ["A", "B", "C"]
    assert first.model.predict(matrix).shape == (6,)
    assert linear_svm_decision_scores(first.model, matrix).shape == (6, 3)
    assert set(first.model.predict(matrix)).issubset(first.model.classes_)
    np.testing.assert_array_equal(
        first.model.predict(matrix), second.model.predict(matrix)
    )
    np.testing.assert_allclose(
        linear_svm_decision_scores(first.model, matrix),
        linear_svm_decision_scores(second.model, matrix),
    )


def test_binary_linear_svm_scores_are_explicitly_oriented_by_class():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        ["apple red", "apple sweet", "banana yellow", "banana ripe"]
    )
    model = train_linear_svm(matrix, ["A", "A", "B", "B"]).model
    scores = linear_svm_decision_scores(model, matrix)
    assert scores.shape == (4, 2)
    np.testing.assert_allclose(scores[:, 0], -scores[:, 1])
    predicted_indexes = scores.argmax(axis=1)
    np.testing.assert_array_equal(model.classes_[predicted_indexes], model.predict(matrix))


def test_prediction_rows_hold_raw_scores_and_shared_evaluation():
    splits = frozen_splits()
    features = build_tfidf_features(splits, TfidfConfig(min_df=1))
    model = train_linear_svm(
        features.train, [record.label for record in splits.train]
    ).model
    rows, timing = predict_linear_svm_records(
        model,
        features.test,
        splits.test,
        model.classes_,
        warmup_iterations=1,
        measured_iterations=2,
    )
    assert len(rows) == len(splits.test)
    assert rows[0]["record_id"] == splits.test[0].record_id
    assert rows[0]["actual_label"] == splits.test[0].label
    assert rows[0]["predicted_label"] in model.classes_
    assert list(rows[0]["decision_scores"]) == list(model.classes_)
    assert rows[0]["scores_are_probabilities"] is False
    assert rows[0]["score_type"] == "raw_linear_svm_decision_function"
    assert timing["warmup_iterations"] == 1
    assert timing["measured_iterations"] == 2
    assert timing["record_count"] == len(splits.test)
    metrics, matrix = evaluate_predictions(
        [row["actual_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        list(model.classes_),
    )
    assert metrics["class_order"] == list(model.classes_)
    assert matrix["row_labels"] == matrix["column_labels"] == list(model.classes_)


def test_linear_svm_workflow_reuses_features_and_verifies_reload(
    tmp_path, monkeypatch
):
    splits, features, split_dir, feature_dir = persist_inputs(tmp_path)
    source_bytes = {
        path: path.read_bytes()
        for directory in (split_dir, feature_dir)
        for path in directory.iterdir()
    }

    def forbid(*args, **kwargs):
        raise AssertionError("TF-IDF must not refit for Linear SVM")

    monkeypatch.setattr(TfidfVectorizer, "fit", forbid)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", forbid)
    output = tmp_path / "linear_svm"
    manifest = run_linear_svm_baseline(split_dir, feature_dir, output)
    assert manifest["reload_verified"] is True
    assert manifest["model_type"] == "linear_svm"
    assert manifest["decision_scores_are_probabilities"] is False
    assert manifest["model_size_bytes"] == (output / "model.joblib").stat().st_size
    assert manifest["dataset_version"] == splits.manifest.dataset_version
    model = load_linear_svm(output / "model.joblib")
    for name in ("validation", "test"):
        rows = [
            json.loads(line)
            for line in (output / f"{name}_predictions.jsonl").read_text().splitlines()
        ]
        np.testing.assert_array_equal(
            [row["predicted_label"] for row in rows],
            model.predict(getattr(features, name)),
        )
        saved_scores = np.asarray(
            [
                [row["decision_scores"][label] for label in model.classes_]
                for row in rows
            ]
        )
        np.testing.assert_allclose(
            saved_scores,
            linear_svm_decision_scores(model, getattr(features, name)),
            rtol=1e-12,
            atol=1e-12,
        )
        assert [row["record_id"] for row in rows] == [
            record.record_id for record in getattr(splits, name)
        ]
    assert all(path.read_bytes() == value for path, value in source_bytes.items())
    for name in (
        "config.json",
        "metrics.json",
        "confusion_matrix.json",
        "experiment_manifest.json",
    ):
        json.loads((output / name).read_text())


def test_linear_svm_holdout_changes_do_not_influence_training(tmp_path):
    _, features, split_dir, feature_dir = persist_inputs(tmp_path)
    run_linear_svm_baseline(split_dir, feature_dir, tmp_path / "first")
    first = load_linear_svm(tmp_path / "first/model.joblib")
    second = train_linear_svm(
        features.train,
        ["Incident", "Request", "Incident", "Request"],
    ).model
    np.testing.assert_allclose(first.coef_, second.coef_)
    np.testing.assert_allclose(first.intercept_, second.intercept_)

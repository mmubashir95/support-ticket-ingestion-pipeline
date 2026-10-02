"""Focused Phase 2.4 training, evaluation, persistence and leakage checks."""
import json

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer

from ticket_classification.classical import (
    LogisticRegressionConfig, coefficient_feature_mapping, measure_cpu_inference,
    predict_records, train_logistic_regression, validate_features,
)
from ticket_classification.classical_outputs import load_logistic_regression, run_logistic_regression_baseline
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.features import TfidfConfig, build_tfidf_features
from ticket_classification.feature_outputs import write_feature_artifacts
from ticket_classification.split_outputs import serialize_classification_records, split_manifest_to_json
from test_features import frozen_splits


@pytest.mark.parametrize("values", [
    {"C": 0}, {"C": -1}, {"C": float("inf")}, {"C": float("nan")},
    {"max_iterations": 0}, {"max_iterations": -1}, {"max_iterations": True},
    {"solver": "liblinear"}, {"penalty": "l1"}, {"class_weight": "balanced"},
    {"class_weight": {"Incident": 2}}, {"random_seed": -1}, {"unknown": 1},
])
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        LogisticRegressionConfig(**values)


def test_configuration_roundtrip():
    config = LogisticRegressionConfig()
    assert config.model_dump() == dict(solver="lbfgs", penalty="l2", C=1.0,
                                      max_iterations=1000, class_weight=None, random_seed=42)
    assert LogisticRegressionConfig.model_validate_json(config.model_dump_json()) == config


def test_training_prediction_probability_reproducibility_and_binary_coefficients():
    splits = frozen_splits()
    features = build_tfidf_features(splits, TfidfConfig(min_df=1))
    labels = [record.label for record in splits.train]
    first = train_logistic_regression(features.train, labels)
    second = train_logistic_regression(features.train, labels)
    assert list(first.model.classes_) == ["Incident", "Request"]
    assert first.training_time >= 0
    np.testing.assert_allclose(first.model.coef_, second.model.coef_)
    rows = predict_records(first.model, features.test, splits.test, first.model.classes_)
    assert len(rows) == len(splits.test)
    assert rows[0]["record_id"] == splits.test[0].record_id
    assert rows[0]["predicted_label"] in first.model.classes_
    assert rows[0]["probability_class_order"] == list(first.model.classes_)
    assert sum(rows[0]["probabilities"]) == pytest.approx(1)
    assert len(rows[0]["probabilities"]) == 2
    mapping = coefficient_feature_mapping(first.model, features.vectorizer)
    feature = features.vectorizer.get_feature_names_out()[0]
    assert mapping["Request"][feature] == first.model.coef_[0, 0]
    assert mapping["Incident"][feature] == -first.model.coef_[0, 0]
    with pytest.raises(ValueError, match="class order"):
        predict_records(first.model, features.test, splits.test, ["Request", "Incident"])


def test_shared_cpu_timing_covers_text_to_label_mechanics_only():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["apple apple", "banana banana", "cherry cherry"])
    model = train_logistic_regression(matrix, ["A", "B", "C"]).model
    texts = ["apple", "banana", "cherry", "apple banana"]
    calls = []
    original = vectorizer.transform
    vectorizer.transform = lambda batch: calls.append(len(batch)) or original(batch)
    timing = measure_cpu_inference(model, vectorizer, texts, warmup_iterations=1,
                                   measured_iterations=3, single_record_samples=2)
    assert timing["protocol"] == "text_to_label_v1"
    assert "TF-IDF transform" in timing["includes"]
    assert timing["statistic"] == "median"
    assert (timing["record_count"], timing["single_record_samples"]) == (4, 2)
    assert timing["seconds_per_record"] == pytest.approx(timing["batch_seconds"] / 4)
    assert timing["records_per_second"] == pytest.approx(4 / timing["batch_seconds"])
    assert all(timing[key] > 0 for key in ("batch_seconds", "predict_only_seconds_per_record",
                                           "single_record_median_seconds"))
    # 1 untimed transform + 1 warm-up + 3 measured batches, then 1 warm-up + 2 single-ticket calls.
    assert calls == [4, 4, 4, 4, 4, 1, 1, 1]
    with pytest.raises(ValueError, match="at least one"):
        measure_cpu_inference(model, vectorizer, [])
    with pytest.raises(ValueError, match="measurement"):
        measure_cpu_inference(model, vectorizer, texts, measured_iterations=0)


def test_multiclass_training_and_feature_mapping():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["apple apple", "banana banana", "cherry cherry"])
    model = train_logistic_regression(matrix, ["A", "B", "C"]).model
    assert model.predict_proba(matrix).shape == (3, 3)
    mapping = coefficient_feature_mapping(model, vectorizer)
    for index, label in enumerate(model.classes_):
        np.testing.assert_allclose(list(mapping[label].values()), model.coef_[index])


def test_dimension_and_label_validation():
    features = build_tfidf_features(frozen_splits(), TfidfConfig(min_df=1))
    with pytest.raises(ValueError, match="dimensions"):
        validate_features(features.train, ["A"], features.train.shape[1])
    with pytest.raises(ValueError, match="dimensions"):
        validate_features(features.train, ["A"] * 4, features.train.shape[1] + 1)
    with pytest.raises(ValueError, match="sparse"):
        validate_features(features.train.toarray(), ["A"] * 4, features.train.shape[1])
    with pytest.raises(ValueError, match="two classes"):
        train_logistic_regression(features.train, ["A"] * 4)


def test_convergence_warning_is_not_swallowed():
    features = build_tfidf_features(frozen_splits(), TfidfConfig(min_df=1))
    with pytest.raises(ConvergenceWarning):
        train_logistic_regression(features.train, ["A", "B", "A", "B"],
                                  LogisticRegressionConfig(max_iterations=1))


def test_known_metrics_and_missing_predicted_class():
    metrics, matrix = evaluate_predictions(["A", "A", "B", "C"], ["A", "B", "B", "B"], ["A", "B", "C"])
    assert metrics["accuracy"] == .5
    assert metrics["macro_precision"] == pytest.approx(4 / 9)
    assert metrics["macro_recall"] == pytest.approx(.5)
    assert metrics["macro_f1"] == pytest.approx(7 / 18)
    assert metrics["weighted_f1"] == pytest.approx(11 / 24)
    assert metrics["per_class"]["C"] == dict(precision=0, recall=0, f1=0, support=1)
    assert metrics["per_class"]["B"]["precision"] == pytest.approx(1 / 3)
    assert matrix["row_labels"] == matrix["column_labels"] == ["A", "B", "C"]
    assert matrix["values"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    with pytest.raises(ValueError, match="unknown"):
        evaluate_predictions(["X"], ["A"], ["A", "B"])


def persist_inputs(tmp_path):
    splits = frozen_splits()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for name in ("train", "validation", "test"):
        (split_dir / f"{name}.jsonl").write_text(serialize_classification_records(getattr(splits, name)))
    (split_dir / "split_manifest.json").write_text(split_manifest_to_json(splits.manifest))
    features = build_tfidf_features(splits, TfidfConfig(min_df=1))
    feature_dir = tmp_path / "tfidf"
    write_feature_artifacts(splits, features, TfidfConfig(min_df=1), feature_dir)
    return splits, features, split_dir, feature_dir


def test_workflow_no_refit_artifacts_reload_and_source_integrity(tmp_path, monkeypatch):
    splits, features, split_dir, feature_dir = persist_inputs(tmp_path)
    original = {path: path.read_bytes() for directory in (split_dir, feature_dir) for path in directory.iterdir()}
    def forbid(*args, **kwargs):
        raise AssertionError("TF-IDF must not refit")
    monkeypatch.setattr(TfidfVectorizer, "fit", forbid)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", forbid)
    output = tmp_path / "lr"
    manifest = run_logistic_regression_baseline(split_dir, feature_dir, output)
    assert manifest["reload_verified"]
    assert manifest["model_size_bytes"] == (output / "model.joblib").stat().st_size
    assert manifest["dataset_version"] == splits.manifest.dataset_version
    model = load_logistic_regression(output / "model.joblib")
    for name in ("validation", "test"):
        rows = [json.loads(line) for line in (output / f"{name}_predictions.jsonl").read_text().splitlines()]
        np.testing.assert_array_equal([row["predicted_label"] for row in rows], model.predict(getattr(features, name)))
        np.testing.assert_allclose([row["probabilities"] for row in rows], model.predict_proba(getattr(features, name)), rtol=1e-12, atol=1e-12)
        assert [row["record_id"] for row in rows] == [record.record_id for record in getattr(splits, name)]
    assert all(path.read_bytes() == value for path, value in original.items())
    for name in ("config.json", "metrics.json", "confusion_matrix.json", "coefficients.json", "experiment_manifest.json"):
        json.loads((output / name).read_text())


def test_workflow_rejects_wrong_feature_rows(tmp_path):
    _, features, split_dir, feature_dir = persist_inputs(tmp_path)
    from scipy.sparse import save_npz
    save_npz(feature_dir / "train_features.npz", features.train[::-1])
    with pytest.raises(ValueError, match="transformations"):
        run_logistic_regression_baseline(split_dir, feature_dir, tmp_path / "lr")


def test_accepted_records_to_persisted_baseline(tmp_path):
    from test_dataset_audit import ticket, manifest
    from ticket_classification.workflow import run_classification_dataset_split, run_tfidf_feature_pipeline
    tickets = [ticket(subject=f"{label} ticket {index}", message=f"{label} assistance {index}", ticket_type=label)
               for label in ("Change", "Incident", "Request") for index in range(20)]
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text("".join(item.model_dump_json() + "\n" for item in tickets))
    source_manifest = tmp_path / "dataset_manifest.json"
    source_manifest.write_text(manifest(tickets).model_dump_json())
    split_dir, feature_dir = tmp_path / "splits", tmp_path / "tfidf"
    run_classification_dataset_split(accepted, source_manifest, split_dir)
    run_tfidf_feature_pipeline(split_dir, feature_dir)
    result = run_logistic_regression_baseline(split_dir, feature_dir, tmp_path / "lr")
    assert result["class_order"] == ["Change", "Incident", "Request"]
    assert result["reload_verified"]
    assert set(result["evaluation_metrics"]) == {"validation", "test"}


def test_holdout_changes_do_not_influence_training(tmp_path):
    splits, features, split_dir, feature_dir = persist_inputs(tmp_path)
    run_logistic_regression_baseline(split_dir, feature_dir, tmp_path / "first")
    changed = type(splits)(train=splits.train, manifest=splits.manifest,
                          validation=[splits.validation[0].model_copy(update={"text": "new holdout text"})],
                          test=[splits.test[0].model_copy(update={"text": "different unseen words"})])
    updated_features = type(features)(vectorizer=features.vectorizer, train=features.train,
        validation=features.vectorizer.transform([record.text for record in changed.validation]).tocsr(),
        test=features.vectorizer.transform([record.text for record in changed.test]).tocsr())
    for name in ("validation", "test"):
        (split_dir / f"{name}.jsonl").write_text(serialize_classification_records(getattr(changed, name)))
    write_feature_artifacts(changed, updated_features, TfidfConfig(min_df=1), feature_dir)
    run_logistic_regression_baseline(split_dir, feature_dir, tmp_path / "second")
    one = load_logistic_regression(tmp_path / "first/model.joblib")
    two = load_logistic_regression(tmp_path / "second/model.joblib")
    np.testing.assert_array_equal(one.coef_, two.coef_)
    np.testing.assert_array_equal(one.intercept_, two.intercept_)

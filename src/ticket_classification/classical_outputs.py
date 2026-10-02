"""Load frozen features, train one baseline, persist and verify its artifacts."""

import json
from collections import Counter
from dataclasses import dataclass
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ticket_classification.classical import (
    LinearSVMConfig, LogisticRegressionConfig, coefficient_feature_mapping,
    linear_svm_decision_scores, measure_cpu_inference, predict_linear_svm_records,
    predict_records, train_linear_svm, train_logistic_regression, validate_features,
)
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.feature_outputs import (
    build_feature_config, build_feature_metadata, load_feature_config,
    load_feature_metadata, load_sparse_features, load_vectorizer,
    _atomic_joblib_dump,
)
from ticket_classification.features import (
    TfidfConfig, TfidfFeatureResult, build_classification_text, load_frozen_splits,
)
from ticket_classification.split_outputs import _atomic_write_text
from ticket_pipeline.versioning import fingerprint_value


def load_logistic_regression(path):
    from sklearn.linear_model import LogisticRegression
    model = joblib.load(path)
    if not isinstance(model, LogisticRegression) or not hasattr(model, "classes_"):
        raise ValueError("artifact must be a fitted Logistic Regression model")
    return model


def load_linear_svm(path):
    from sklearn.svm import LinearSVC
    model = joblib.load(path)
    if not isinstance(model, LinearSVC) or not hasattr(model, "classes_"):
        raise ValueError("artifact must be a fitted Linear SVM model")
    return model


def _write_json(path, value):
    _atomic_write_text(path, json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class ClassicalInputs:
    splits: object
    feature_dir: Path
    vectorizer: object
    feature_config: object
    feature_metadata: object
    matrices: dict
    classes: list[str]


def _load_validated_classical_inputs(split_dir, feature_dir, *, phase):
    """Load and validate the one shared Phase 2.2/2.3 classical input set."""
    splits = load_frozen_splits(split_dir)
    if splits.manifest.target != "ticket_type":
        raise ValueError(f"{phase} target must be ticket_type")
    directory = Path(feature_dir)
    vectorizer = load_vectorizer(directory / "vectorizer.joblib")
    feature_config = load_feature_config(directory / "feature_config.json")
    metadata = load_feature_metadata(directory / "feature_metadata.json")
    matrices = {
        name: load_sparse_features(directory / f"{name}_features.npz")
        for name in ("train", "validation", "test")
    }
    result = TfidfFeatureResult(vectorizer=vectorizer, **matrices)
    config_values = feature_config.model_dump()
    tfidf_config = TfidfConfig(
        **{key: config_values[key] for key in TfidfConfig.model_fields}
    )
    expected = build_feature_metadata(
        splits, result, build_feature_config(tfidf_config)
    )
    if expected != metadata:
        raise ValueError("TF-IDF metadata does not match frozen split/features")
    classes = sorted(splits.manifest.class_counts_full)
    for name, matrix in matrices.items():
        records = getattr(splits, name)
        labels = [record.label for record in records]
        validate_features(matrix, labels, metadata.feature_count)
        if any(not isinstance(label, str) or label not in classes for label in labels):
            raise ValueError("labels must belong to the frozen single-label target set")
        if Counter(labels) != getattr(splits.manifest, f"class_counts_{name}"):
            raise ValueError(f"{name} labels differ from frozen manifest class counts")
        transformed = vectorizer.transform([record.text for record in records]).tocsr()
        if transformed.shape != matrix.shape:
            raise ValueError(f"{name} transformed feature dimensions differ")
        difference = transformed - matrix
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
            raise ValueError(f"{name} features differ from frozen record transformations")
    if set(record.label for record in splits.train) != set(classes):
        raise ValueError("training split must contain every frozen target class")
    return ClassicalInputs(
        splits=splits,
        feature_dir=directory,
        vectorizer=vectorizer,
        feature_config=feature_config,
        feature_metadata=metadata,
        matrices=matrices,
        classes=classes,
    )


def run_logistic_regression_baseline(split_dir, feature_dir, output_dir, *, config=None):
    """Consume Phase 2.2/2.3 artifacts unchanged; never fit a vectorizer."""
    settings = config or LogisticRegressionConfig()
    inputs = _load_validated_classical_inputs(split_dir, feature_dir, phase="Phase 2.4")
    splits, matrices, classes = inputs.splits, inputs.matrices, inputs.classes
    directory, vectorizer = inputs.feature_dir, inputs.vectorizer
    feature_config, metadata = inputs.feature_config, inputs.feature_metadata
    trained = train_logistic_regression(
        matrices["train"], [record.label for record in splits.train], settings,
    )
    model = trained.model
    output = Path(output_dir)
    if output.resolve() in (Path(split_dir).resolve(), directory.resolve()):
        raise ValueError("model output directory must differ from source artifacts")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    _atomic_joblib_dump(model_path, model)
    reloaded = load_logistic_regression(model_path)
    metrics, confusion, timings = {}, {}, {}
    for name in ("validation", "test"):
        matrix, records = matrices[name], getattr(splits, name)
        rows = predict_records(model, matrix, records, classes)
        timings[name] = measure_cpu_inference(
            model, vectorizer, [build_classification_text(record) for record in records])
        np.testing.assert_array_equal(model.predict(matrix), reloaded.predict(matrix))
        np.testing.assert_allclose(model.predict_proba(matrix), reloaded.predict_proba(matrix),
                                   rtol=1e-12, atol=1e-12)
        metrics[name], confusion[name] = evaluate_predictions(
            [row["actual_label"] for row in rows],
            [row["predicted_label"] for row in rows], classes,
        )
        _atomic_write_text(output / f"{name}_predictions.jsonl", "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n" for row in rows))
    mapping = coefficient_feature_mapping(model, vectorizer)
    inspection = {}
    for label, weights in mapping.items():
        ordered = sorted(weights.items(), key=lambda item: (item[1], item[0]))
        inspection[label] = {
            "lowest": [{"feature": name, "coefficient": weight} for name, weight in ordered[:10]],
            "highest": [{"feature": name, "coefficient": weight} for name, weight in ordered[-10:][::-1]],
        }
    measurements = {"training_time": trained.training_time, "model_size_bytes": model_path.stat().st_size,
                    "inference_latency": timings}
    identity = {"dataset_version": metadata.dataset_version, "split_version": metadata.split_version,
                "feature_version": metadata.feature_version, "configuration": settings.model_dump(mode="json")}
    filenames = ["model.joblib", "config.json", "metrics.json", "confusion_matrix.json",
                 "validation_predictions.jsonl", "test_predictions.jsonl", "coefficients.json",
                 "experiment_manifest.json"]
    manifest = {
        **identity, **measurements,
        "experiment_id": f"lr_{fingerprint_value(identity)[:12]}",
        "model_type": "logistic_regression", "model_name": "tfidf_logistic_regression_baseline",
        "target": "ticket_type", "task": "single_label_multiclass", "class_order": classes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_artifact_reference": str(directory.resolve()),
        "feature_configuration": feature_config.model_dump(mode="json"),
        "split_artifact_reference": str(Path(split_dir).resolve()),
        "evaluation_metrics": metrics, "reload_verified": True,
        "reload_verification": "All validation/test predictions identical; probabilities allclose rtol=atol=1e-12",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
        "artifacts": {name: str((output / name).resolve()) for name in filenames},
    }
    _write_json(output / "config.json", settings.model_dump(mode="json"))
    _write_json(output / "metrics.json", {**metrics, **measurements})
    _write_json(output / "confusion_matrix.json", confusion)
    _write_json(output / "coefficients.json", {"class_order": classes, "features_per_extreme": 10,
                                              "per_class": inspection})
    _write_json(output / "experiment_manifest.json", manifest)
    return manifest


def run_linear_svm_baseline(split_dir, feature_dir, output_dir, *, config=None):
    """Train and persist LinearSVC using the frozen Phase 2.2/2.3 artifacts."""
    settings = config or LinearSVMConfig()
    inputs = _load_validated_classical_inputs(split_dir, feature_dir, phase="Phase 2.5")
    splits, matrices, classes = inputs.splits, inputs.matrices, inputs.classes
    directory, vectorizer = inputs.feature_dir, inputs.vectorizer
    feature_config, metadata = inputs.feature_config, inputs.feature_metadata

    trained = train_linear_svm(
        matrices["train"],
        [record.label for record in splits.train],
        settings,
    )
    model = trained.model
    output = Path(output_dir)
    if output.resolve() in (Path(split_dir).resolve(), directory.resolve()):
        raise ValueError("model output directory must differ from source artifacts")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    _atomic_joblib_dump(model_path, model)
    reloaded = load_linear_svm(model_path)

    metrics, confusion, timings = {}, {}, {}
    for name in ("validation", "test"):
        matrix, records = matrices[name], getattr(splits, name)
        rows = predict_linear_svm_records(model, matrix, records, classes)
        timings[name] = measure_cpu_inference(
            model, vectorizer, [build_classification_text(record) for record in records]
        )
        np.testing.assert_array_equal(model.predict(matrix), reloaded.predict(matrix))
        np.testing.assert_allclose(
            linear_svm_decision_scores(model, matrix),
            linear_svm_decision_scores(reloaded, matrix),
            rtol=1e-12,
            atol=1e-12,
        )
        metrics[name], confusion[name] = evaluate_predictions(
            [row["actual_label"] for row in rows],
            [row["predicted_label"] for row in rows],
            classes,
        )
        _atomic_write_text(
            output / f"{name}_predictions.jsonl",
            "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ),
        )

    measurements = {
        "training_time": trained.training_time,
        "model_size_bytes": model_path.stat().st_size,
        "inference_latency": timings,
    }
    identity = {
        "dataset_version": metadata.dataset_version,
        "split_version": metadata.split_version,
        "feature_version": metadata.feature_version,
        "configuration": settings.model_dump(mode="json"),
    }
    filenames = [
        "model.joblib",
        "config.json",
        "metrics.json",
        "confusion_matrix.json",
        "validation_predictions.jsonl",
        "test_predictions.jsonl",
        "experiment_manifest.json",
    ]
    manifest = {
        **identity,
        **measurements,
        "experiment_id": f"svm_{fingerprint_value(identity)[:12]}",
        "model_type": "linear_svm",
        "model_name": "tfidf_linear_svm_baseline",
        "target": "ticket_type",
        "task": "single_label_multiclass",
        "class_order": classes,
        "label_mapping": {label: index for index, label in enumerate(classes)},
        "decision_score_type": "raw_linear_svm_decision_function",
        "decision_scores_are_probabilities": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_artifact_reference": str(directory.resolve()),
        "feature_configuration": feature_config.model_dump(mode="json"),
        "split_artifact_reference": str(Path(split_dir).resolve()),
        "evaluation_metrics": metrics,
        "reload_verified": True,
        "reload_verification": (
            "All validation/test predictions identical; raw decision scores "
            "allclose rtol=atol=1e-12"
        ),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "artifacts": {
            name: str((output / name).resolve()) for name in filenames
        },
    }
    _write_json(output / "config.json", settings.model_dump(mode="json"))
    _write_json(output / "metrics.json", {**metrics, **measurements})
    _write_json(output / "confusion_matrix.json", confusion)
    _write_json(output / "experiment_manifest.json", manifest)
    return manifest

"""Explicit, validation-only Phase 2.10 calibration and threshold workflow."""

from __future__ import annotations

from collections import Counter
import json
import platform
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import train_test_split

from ticket_classification.calibration import (
    SigmoidScoreCalibrator,
    TemperatureScaler,
    evaluate_calibration,
    fit_sigmoid_score_calibration,
    fit_temperature_scaling,
    validate_probabilities,
)
from ticket_classification.classical_outputs import load_linear_svm, load_logistic_regression
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.feature_outputs import load_sparse_features
from ticket_classification.split_outputs import (
    _atomic_write_text,
    load_classification_records,
    load_split_manifest,
)
from ticket_classification.thresholds import (
    DEFAULT_THRESHOLDS,
    apply_threshold,
    evaluate_thresholds,
    select_threshold_for_accuracy,
)
from ticket_pipeline.versioning import fingerprint_value


ModelName = Literal["transformer", "logistic_regression", "linear_svm"]
MODEL_NAMES: tuple[ModelName, ...] = (
    "transformer",
    "logistic_regression",
    "linear_svm",
)
LABEL_ORDER = ("Change", "Incident", "Problem", "Request")


def _read_json(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {source}")
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"prediction row must be an object: {source}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        target,
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
    )


def _environment() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "unavailable"
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "unavailable"
    return versions


def verify_phase29_policy(run_root: str | Path) -> dict[str, Any]:
    """Validate the persisted selected weighting policy and source identities."""

    root = Path(run_root)
    policy_path = root / "class_weighting" / "final_weighting_policy.json"
    policy = _read_json(policy_path)
    required = {
        "logistic_regression": "balanced",
        "linear_svm": "balanced",
        "distilbert": "none",
    }
    for model, expected in required.items():
        details = policy.get(model)
        if not isinstance(details, dict) or details.get("status") != "SELECTED":
            raise ValueError(f"Phase 2.9 policy is not selected for {model}")
        if details.get("selected_weighting_policy") != expected:
            raise ValueError(
                f"Phase 2.9 {model} selection contradicts the expected persisted policy"
            )
    transformer_manifest = _read_json(root / "transformer_training" / "training_manifest.json")
    transformer_checkpoint = _read_json(root / "transformer_training" / "best_checkpoint.json")
    feature_metadata = _read_json(root / "tfidf" / "feature_metadata.json")
    if (
        feature_metadata.get("dataset_version") != transformer_manifest["dataset_version"]
        or feature_metadata.get("split_version") != transformer_manifest["split_version"]
    ):
        raise ValueError("TF-IDF and Transformer experiment identities differ")
    selected_reference = policy["distilbert"].get("selected_checkpoint_reference")
    if transformer_checkpoint.get("path") != selected_reference:
        raise ValueError("Phase 2.9 Transformer checkpoint reference is incompatible")
    if transformer_manifest.get("class_weighting") not in ("none", "disabled"):
        raise ValueError("selected Transformer experiment is not unweighted")
    classical: dict[str, Any] = {}
    for name in ("logistic_regression", "linear_svm"):
        manifest_path = root / "class_weighting" / name / "experiment_manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("configuration", {}).get("class_weight") != "balanced":
            raise ValueError(f"selected {name} artifact is not balanced")
        if manifest.get("test_set_used") is not False:
            raise ValueError(f"selected {name} manifest does not prove test isolation")
        classical[name] = {
            "weighting": "balanced",
            "feature_version": feature_metadata["feature_version"],
            "model_artifact": str(
                root / "class_weighting" / name / "weighted_model.joblib"
            ),
            "prediction_artifact": str(
                root / "class_weighting" / name / "validation_predictions.jsonl"
            ),
            "manifest": str(manifest_path),
        }
    test_artifact_paths = [
        root / "transformer_training" / name
        for name in (
            "test_evaluation_manifest.json",
            "test_metrics.json",
            "test_confusion_matrix.json",
            "test_predictions.jsonl",
        )
    ]
    test_artifacts_present = all(path.is_file() for path in test_artifact_paths)
    status_discrepancies = []
    if (
        policy["distilbert"].get("final_test_evaluation_status") == "PENDING"
        and test_artifacts_present
    ):
        status_discrepancies.append(
            "Phase 2.9 policy marks DistilBERT final test evaluation PENDING, "
            "but the complete test artifact set exists. Phase 2.10 does not read "
            "or use those artifacts for calibration or threshold selection."
        )
    return {
        "policy_artifact": str(policy_path),
        "dataset_version": transformer_manifest["dataset_version"],
        "split_version": transformer_manifest["split_version"],
        "transformer_dataset_version": transformer_manifest[
            "transformer_dataset_version"
        ],
        "feature_version": feature_metadata["feature_version"],
        "label_order": list(LABEL_ORDER),
        "status_discrepancies": status_discrepancies,
        "test_artifacts_detected_but_not_loaded": test_artifacts_present,
        "models": {
            **classical,
            "transformer": {
                "weighting": "none",
                "model_name": transformer_manifest["configuration"]["model_name"],
                "checkpoint": selected_reference,
                "prediction_artifact": str(
                    root / "transformer_training" / "validation_predictions.jsonl"
                ),
                "manifest": str(
                    root / "transformer_training" / "training_manifest.json"
                ),
            },
        },
    }


def _normalize_prediction_rows(
    model: ModelName, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    ids: list[str] = []
    actual: list[str] = []
    predicted: list[str] = []
    probabilities: list[list[float]] | None = []
    logits_or_scores: list[list[float]] | None = []
    for row in rows:
        ids.append(str(row.get("record_id", "")))
        actual.append(str(row.get("true_label", row.get("actual_label", ""))))
        predicted.append(str(row.get("predicted_label", "")))
        if model == "linear_svm":
            order = row.get("decision_score_class_order")
            score_map = row.get("decision_scores")
            if order != list(LABEL_ORDER) or not isinstance(score_map, dict):
                raise ValueError("Linear SVM decision-score schema is incompatible")
            if row.get("scores_are_probabilities") is not False:
                raise ValueError("Linear SVM scores must be explicitly marked non-probabilistic")
            logits_or_scores.append([score_map[label] for label in LABEL_ORDER])
            probabilities = None
        else:
            if (
                model == "logistic_regression"
                and row.get("probability_class_order") != list(LABEL_ORDER)
            ):
                raise ValueError("Logistic Regression probability class order is incompatible")
            probability = row.get("probabilities")
            if not isinstance(probability, list):
                raise ValueError(f"{model} prediction lacks probabilities")
            probabilities.append(probability)
            if model == "transformer":
                expected_true_id = (
                    LABEL_ORDER.index(actual[-1]) if actual[-1] in LABEL_ORDER else None
                )
                expected_predicted_id = (
                    LABEL_ORDER.index(predicted[-1])
                    if predicted[-1] in LABEL_ORDER
                    else None
                )
                if (
                    row.get("true_label_id") != expected_true_id
                    or row.get("predicted_label_id") != expected_predicted_id
                ):
                    raise ValueError("Transformer label IDs differ from frozen label mapping")
                logits = row.get("logits")
                if not isinstance(logits, list):
                    raise ValueError("Transformer prediction lacks logits")
                logits_or_scores.append(logits)
            else:
                logits_or_scores = None
    return {
        "record_ids": ids,
        "actual_labels": actual,
        "predicted_labels": predicted,
        "probabilities": (
            np.asarray(probabilities, dtype=np.float64)
            if probabilities is not None
            else None
        ),
        "logits_or_scores": (
            np.asarray(logits_or_scores, dtype=np.float64)
            if logits_or_scores is not None
            else None
        ),
    }


def load_selected_predictions(
    run_root: str | Path,
    model: ModelName,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = verify_phase29_policy(run_root)
    source = Path(selected["models"][model]["prediction_artifact"])
    generated = (
        Path(output_dir) / "generated_predictions" / f"{model}_validation_predictions.jsonl"
        if output_dir is not None
        else None
    )
    if not source.is_file() and generated is not None and generated.is_file():
        source = generated
    return _normalize_prediction_rows(model, _read_jsonl(source)), selected


def generate_selected_validation_predictions(
    run_root: str | Path,
    output_dir: str | Path,
    model: ModelName,
) -> Path:
    """Regenerate validation predictions from a frozen selected model, never test."""

    root = Path(run_root)
    selected = verify_phase29_policy(root)
    target = (
        Path(output_dir)
        / "generated_predictions"
        / f"{model}_validation_predictions.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if model in ("logistic_regression", "linear_svm"):
        from ticket_classification.classical import (
            predict_linear_svm_records,
            predict_records,
        )

        records = load_classification_records(root / "splits" / "validation.jsonl")
        matrix = load_sparse_features(root / "tfidf" / "validation_features.npz")
        model_path = selected["models"][model]["model_artifact"]
        if model == "logistic_regression":
            estimator = load_logistic_regression(model_path)
            rows = predict_records(estimator, matrix, records, LABEL_ORDER)
        else:
            estimator = load_linear_svm(model_path)
            rows = predict_linear_svm_records(estimator, matrix, records, LABEL_ORDER)
    else:
        from transformers import AutoModelForSequenceClassification

        from ticket_classification.transformer_training import (
            DynamicPaddingCollator,
            TransformerTrainingConfig,
            build_data_loader,
            evaluate_model,
            load_training_data,
            resolve_device,
            set_reproducible_seed,
            validate_model_configuration,
        )

        config = TransformerTrainingConfig.model_validate(
            _read_json(root / "transformer_training" / "training_config.json")
        )
        training = load_training_data(root / "transformer", config)
        set_reproducible_seed(config.random_seed)
        device = resolve_device(config.device_preference)
        estimator = AutoModelForSequenceClassification.from_pretrained(
            selected["models"]["transformer"]["checkpoint"]
        ).to(device)
        validate_model_configuration(estimator, config)
        loader = build_data_loader(
            training.validation,
            config.eval_batch_size,
            DynamicPaddingCollator(training.tokenizer_policy.pad_token_id),
            shuffle=False,
            seed=config.random_seed,
        )
        rows = evaluate_model(
            estimator, loader, device, config.label_order
        ).predictions
    content = "".join(
        json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _atomic_write_text(target, content)
    return target


def audit_prediction_artifacts(
    run_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Audit selected validation artifacts without opening test prediction files."""

    root = Path(run_root)
    selected = verify_phase29_policy(root)
    split_manifest = load_split_manifest(root / "splits" / "split_manifest.json")
    validation_records = load_classification_records(
        root / "splits" / "validation.jsonl"
    )
    expected_ids = split_manifest.validation_record_ids
    expected_labels = {record.record_id: record.label for record in validation_records}
    test_ids = set(split_manifest.test_record_ids)
    model_audits: dict[str, Any] = {}
    for model in MODEL_NAMES:
        source = Path(selected["models"][model]["prediction_artifact"])
        generated = (
            Path(output_dir)
            / "generated_predictions"
            / f"{model}_validation_predictions.jsonl"
        )
        if not source.is_file() and generated.is_file():
            source = generated
        if not source.is_file():
            model_audits[model] = {
                "status": "missing",
                "source": str(source),
                "compatible": False,
                "reason": "selected validation prediction artifact is missing",
            }
            continue
        try:
            raw_rows = _read_jsonl(source)
            normalized = _normalize_prediction_rows(model, raw_rows)
            ids = normalized["record_ids"]
            if not ids or any(not record_id for record_id in ids):
                raise ValueError("record IDs must be nonblank")
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate record IDs")
            if ids != expected_ids:
                raise ValueError("record membership/order differs from frozen validation split")
            if set(ids) & test_ids:
                raise ValueError("test record entered validation predictions")
            if normalized["actual_labels"] != [expected_labels[item] for item in ids]:
                raise ValueError("true labels differ from the frozen validation split")
            if any(label not in LABEL_ORDER for label in normalized["predicted_labels"]):
                raise ValueError("prediction contains an unknown label")
            probabilities = normalized["probabilities"]
            if probabilities is not None:
                validate_probabilities(
                    probabilities, expected_classes=len(LABEL_ORDER)
                )
                argmax_labels = [
                    LABEL_ORDER[index] for index in probabilities.argmax(axis=1)
                ]
                if normalized["predicted_labels"] != argmax_labels:
                    raise ValueError("predicted labels differ from probability argmax")
            scores = normalized["logits_or_scores"]
            if scores is not None:
                if scores.shape != (len(ids), len(LABEL_ORDER)):
                    raise ValueError("logit/score dimensions are incompatible")
                if not np.isfinite(scores).all():
                    raise ValueError("logits/scores contain non-finite values")
                argmax_labels = [LABEL_ORDER[index] for index in scores.argmax(axis=1)]
                if normalized["predicted_labels"] != argmax_labels:
                    raise ValueError("predicted labels differ from logit/score argmax")
            model_audits[model] = {
                "status": "available",
                "compatible": True,
                "source": str(source),
                "prediction_artifact_fingerprint": fingerprint_value(raw_rows),
                "record_count": len(ids),
                "unique_record_ids": True,
                "frozen_validation_membership": True,
                "test_overlap_count": 0,
                "label_order": list(LABEL_ORDER),
                "probabilities": "valid" if probabilities is not None else "not_applicable",
                "logits_or_scores": "valid" if scores is not None else "unavailable",
                "argmax_consistent": True,
            }
        except (ValueError, KeyError, TypeError) as error:
            model_audits[model] = {
                "status": "incompatible",
                "compatible": False,
                "source": str(source),
                "reason": str(error),
            }
    audit = {
        "phase": "2.10",
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "transformer_dataset_version": selected["transformer_dataset_version"],
        "source_split": "validation",
        "test_set_used": False,
        "expected_record_count": len(expected_ids),
        "label_mapping": {label: index for index, label in enumerate(LABEL_ORDER)},
        "phase_2_9_policy": selected,
        "models": model_audits,
        "audit_passed": all(item["compatible"] for item in model_audits.values()),
    }
    audit["audit_id"] = "prediction_audit_" + fingerprint_value(audit)[:12]
    _write_json(Path(output_dir) / "prediction_artifact_audit.json", audit)
    return audit


def create_calibration_split(
    run_root: str | Path, output_dir: str | Path, *, seed: int = 42
) -> dict[str, Any]:
    """Split validation deterministically; never modify the frozen split."""

    root = Path(run_root)
    selected = verify_phase29_policy(root)
    frozen = load_split_manifest(root / "splits" / "split_manifest.json")
    records = load_classification_records(root / "splits" / "validation.jsonl")
    ids = [record.record_id for record in records]
    labels = [record.label for record in records]
    calibration_ids, threshold_ids = train_test_split(
        ids,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    calibration_set, threshold_set = set(calibration_ids), set(threshold_ids)
    if calibration_set & threshold_set or calibration_set | threshold_set != set(ids):
        raise RuntimeError("internal validation subsets are not a disjoint partition")
    if (calibration_set | threshold_set) & set(frozen.test_record_ids):
        raise RuntimeError("test records entered the Phase 2.10 development subsets")
    labels_by_id = {record.record_id: record.label for record in records}
    manifest = {
        "phase": "2.10",
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "source_split": "validation",
        "seed": seed,
        "stratified_by": "true_label",
        "calibration_fitting_count": len(calibration_ids),
        "threshold_selection_count": len(threshold_ids),
        "calibration_class_counts": dict(
            sorted(Counter(labels_by_id[item] for item in calibration_ids).items())
        ),
        "threshold_class_counts": dict(
            sorted(Counter(labels_by_id[item] for item in threshold_ids).items())
        ),
        "subsets_disjoint": True,
        "union_matches_validation": True,
        "test_overlap_count": 0,
        "base_split_modified": False,
        "known_limitation": (
            "The original validation split influenced checkpoint/model selection. "
            "These internal subsets reduce direct fitting/selection leakage but are not "
            "a fully independent evaluation set."
        ),
    }
    identity = {
        "dataset_version": manifest["dataset_version"],
        "split_version": manifest["split_version"],
        "seed": seed,
        "calibration_record_ids": calibration_ids,
        "threshold_selection_record_ids": threshold_ids,
    }
    manifest["calibration_split_id"] = "calibration_split_" + fingerprint_value(identity)[:12]
    output = Path(output_dir)
    _write_json(output / "calibration_split_manifest.json", manifest)
    _write_json(
        output / "calibration_record_ids.json",
        {
            "calibration_split_id": manifest["calibration_split_id"],
            "source_split": "validation",
            "record_ids": calibration_ids,
        },
    )
    _write_json(
        output / "threshold_selection_record_ids.json",
        {
            "calibration_split_id": manifest["calibration_split_id"],
            "source_split": "validation",
            "record_ids": threshold_ids,
        },
    )
    return manifest


def _load_subset_ids(output_dir: str | Path) -> tuple[set[str], set[str], dict[str, Any]]:
    output = Path(output_dir)
    manifest = _read_json(output / "calibration_split_manifest.json")
    calibration = set(_read_json(output / "calibration_record_ids.json")["record_ids"])
    threshold = set(
        _read_json(output / "threshold_selection_record_ids.json")["record_ids"]
    )
    if calibration & threshold:
        raise ValueError("persisted Phase 2.10 subsets overlap")
    return calibration, threshold, manifest


def _subset(normalized: dict[str, Any], ids: set[str]) -> dict[str, Any]:
    indexes = [
        index
        for index, record_id in enumerate(normalized["record_ids"])
        if record_id in ids
    ]
    if len(indexes) != len(ids):
        raise ValueError("prediction artifact does not contain every requested subset record")
    result: dict[str, Any] = {}
    for key, value in normalized.items():
        if value is None:
            result[key] = None
        elif isinstance(value, np.ndarray):
            result[key] = value[indexes]
        else:
            result[key] = [value[index] for index in indexes]
    return result


def _label_ids(labels: Sequence[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(LABEL_ORDER)}
    if any(label not in mapping for label in labels):
        raise ValueError("labels differ from the frozen label mapping")
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def _classification_summary(probabilities: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    predicted = [LABEL_ORDER[index] for index in probabilities.argmax(axis=1)]
    return evaluate_predictions(list(labels), predicted, list(LABEL_ORDER))[0]


def _confidence_summary(
    probabilities: np.ndarray, labels: Sequence[str], *, bins: int
) -> dict[str, Any]:
    label_ids = _label_ids(labels)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == label_ids
    reliability = evaluate_calibration(probabilities, label_ids, bins=bins)[
        "reliability_bins"
    ]
    per_class: dict[str, Any] = {}
    for class_id, label in enumerate(LABEL_ORDER):
        actual = label_ids == class_id
        wrong = actual & ~correct
        per_class[label] = {
            "count": int(actual.sum()),
            "mean_confidence": float(confidence[actual].mean()) if actual.any() else None,
            "incorrect_count": int(wrong.sum()),
            "incorrect_mean_confidence": (
                float(confidence[wrong].mean()) if wrong.any() else None
            ),
        }
    incident, problem = LABEL_ORDER.index("Incident"), LABEL_ORDER.index("Problem")
    incident_problem_confusions = int(
        (((label_ids == incident) & (predicted == problem))
         | ((label_ids == problem) & (predicted == incident))).sum()
    )
    return {
        "record_count": probabilities.shape[0],
        "mean_confidence": float(confidence.mean()),
        "correct_mean_confidence": float(confidence[correct].mean()),
        "incorrect_mean_confidence": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
        "high_confidence_error_count_at_0_90": int(((~correct) & (confidence >= 0.9)).sum()),
        "incident_problem_confusion_count": incident_problem_confusions,
        "confidence_bins": reliability,
        "per_actual_class": per_class,
    }


def analyze_uncalibrated_models(
    run_root: str | Path,
    output_dir: str | Path,
    *,
    bins: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Create development subsets and baseline validation calibration evidence."""

    audit = audit_prediction_artifacts(run_root, output_dir)
    if not audit["audit_passed"]:
        raise ValueError("prediction artifact audit failed; inspect its persisted reasons")
    split_manifest = create_calibration_split(run_root, output_dir, seed=seed)
    comparison: dict[str, Any] = {}
    confidence: dict[str, Any] = {}
    for model in MODEL_NAMES:
        normalized, _ = load_selected_predictions(run_root, model, output_dir)
        if normalized["probabilities"] is None:
            comparison[model] = {
                "status": "not_applicable",
                "reason": "raw Linear SVM decision scores are not probabilities",
                "ece": None,
                "multiclass_brier_score": None,
                "negative_log_likelihood": None,
            }
            confidence[model] = {
                "status": "unavailable",
                "reason": "probability calibration is required before confidence analysis",
            }
            continue
        label_ids = _label_ids(normalized["actual_labels"])
        comparison[model] = {
            "status": "measured",
            **evaluate_calibration(
                normalized["probabilities"], label_ids, bins=bins
            ),
            "classification": _classification_summary(
                normalized["probabilities"], normalized["actual_labels"]
            ),
        }
        confidence[model] = {
            "status": "measured",
            **_confidence_summary(
                normalized["probabilities"], normalized["actual_labels"], bins=bins
            ),
        }
    result = {
        "phase": "2.10",
        "dataset_version": audit["dataset_version"],
        "split_version": audit["split_version"],
        "source_split": "validation",
        "test_set_used": False,
        "bins": bins,
        "calibration_split_id": split_manifest["calibration_split_id"],
        "models": comparison,
        "environment": _environment(),
    }
    output = Path(output_dir)
    _write_json(output / "uncalibrated_model_comparison.json", result)
    confidence_artifact = {
        "dataset_version": audit["dataset_version"],
        "split_version": audit["split_version"],
        "source_split": "validation",
        "models": confidence,
    }
    _write_json(output / "confidence_analysis.json", confidence_artifact)
    _atomic_write_text(
        output / "confidence_analysis.md",
        render_confidence_analysis(confidence_artifact),
    )
    return result


def render_confidence_analysis(artifact: dict[str, Any]) -> str:
    lines = [
        "# Phase 2.10 Confidence Analysis",
        "",
        f"Dataset `{artifact['dataset_version']}`, split `{artifact['split_version']}`.",
        "This report uses validation predictions only; no test predictions are loaded.",
        "",
    ]
    for model, details in artifact["models"].items():
        lines.extend([f"## {model}", ""])
        if details["status"] != "measured":
            lines.extend([f"**{details['status']}** — {details['reason']}", ""])
            continue
        lines.extend(
            [
                f"- Mean confidence: {details['mean_confidence']:.6f}",
                f"- Correct-prediction mean confidence: {details['correct_mean_confidence']:.6f}",
                f"- Incorrect-prediction mean confidence: {details['incorrect_mean_confidence']:.6f}",
                f"- Errors with confidence >= 0.90: {details['high_confidence_error_count_at_0_90']}",
                f"- Incident/Problem confusions: {details['incident_problem_confusion_count']}",
                "",
            ]
        )
    return "\n".join(lines)


def fit_model_calibration(
    run_root: str | Path,
    output_dir: str | Path,
    model: ModelName,
    *,
    bins: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit calibration subset only and evaluate threshold-selection subset only."""

    calibration_ids, threshold_ids, split_manifest = _load_subset_ids(output_dir)
    normalized, selected = load_selected_predictions(run_root, model, output_dir)
    fitting = _subset(normalized, calibration_ids)
    evaluation = _subset(normalized, threshold_ids)
    fitting_labels = _label_ids(fitting["actual_labels"])
    evaluation_labels = _label_ids(evaluation["actual_labels"])
    if model == "transformer":
        scaler = fit_temperature_scaling(
            fitting["logits_or_scores"], fitting_labels
        )
        original_probabilities = evaluation["probabilities"]
        calibrated_probabilities = scaler.transform(evaluation["logits_or_scores"])
        parameter_artifact = {
            **scaler.to_dict(),
            "model": model,
            "fitted_on": "calibration_fitting",
            "record_count": len(calibration_ids),
            "calibration_split_id": split_manifest["calibration_split_id"],
            "base_model_modified": False,
            "source_logits_preserved": True,
        }
        parameter_name = "temperature_scaling.json"
        comparison_name = "distilbert_calibration_comparison.json"
    elif model == "logistic_regression":
        original_probabilities = fitting["probabilities"]
        fitting_scores = np.log(
            np.clip(original_probabilities, np.finfo(float).tiny, 1.0)
        )
        calibrator = fit_sigmoid_score_calibration(
            fitting_scores, fitting_labels, LABEL_ORDER, random_seed=seed
        )
        original_probabilities = evaluation["probabilities"]
        evaluation_scores = np.log(
            np.clip(original_probabilities, np.finfo(float).tiny, 1.0)
        )
        calibrated_probabilities = calibrator.transform(evaluation_scores)
        parameter_artifact = {
            **calibrator.to_dict(),
            "model": model,
            "input_score_type": "log_base_predict_proba",
            "fitted_on": "calibration_fitting",
            "record_count": len(calibration_ids),
            "calibration_split_id": split_manifest["calibration_split_id"],
            "base_model_modified": False,
        }
        parameter_name = "logistic_regression_sigmoid_scaling.json"
        comparison_name = "logistic_regression_calibration_comparison.json"
    else:
        calibrator = fit_sigmoid_score_calibration(
            fitting["logits_or_scores"],
            fitting_labels,
            LABEL_ORDER,
            random_seed=seed,
        )
        original_probabilities = None
        calibrated_probabilities = calibrator.transform(
            evaluation["logits_or_scores"]
        )
        parameter_artifact = {
            **calibrator.to_dict(),
            "model": model,
            "input_score_type": "raw_linear_svm_decision_function",
            "fitted_on": "calibration_fitting",
            "record_count": len(calibration_ids),
            "calibration_split_id": split_manifest["calibration_split_id"],
            "base_model_modified": False,
        }
        parameter_name = "linear_svm_sigmoid_calibration.json"
        comparison_name = "linear_svm_calibration_comparison.json"
    calibrated_metrics = evaluate_calibration(
        calibrated_probabilities, evaluation_labels, bins=bins
    )
    calibrated_classification = _classification_summary(
        calibrated_probabilities, evaluation["actual_labels"]
    )
    original_metrics = (
        evaluate_calibration(original_probabilities, evaluation_labels, bins=bins)
        if original_probabilities is not None
        else None
    )
    original_classification = (
        _classification_summary(original_probabilities, evaluation["actual_labels"])
        if original_probabilities is not None
        else _classification_summary_from_labels(
            evaluation["predicted_labels"], evaluation["actual_labels"]
        )
    )
    decision = _calibration_decision(original_metrics, calibrated_metrics)
    comparison = {
        "phase": "2.10",
        "model": model,
        "model_identity": selected["models"][model],
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "calibration_split_id": split_manifest["calibration_split_id"],
        "fitting_subset": "calibration_fitting",
        "evaluation_subset": "threshold_selection",
        "fitting_record_count": len(calibration_ids),
        "evaluation_record_count": len(threshold_ids),
        "test_set_used": False,
        "parameter_artifact": str(Path(output_dir) / parameter_name),
        "uncalibrated": {
            "probability_metrics": original_metrics,
            "classification": original_classification,
        },
        "calibrated": {
            "probability_metrics": calibrated_metrics,
            "classification": calibrated_classification,
        },
        "differences_calibrated_minus_uncalibrated": (
            _metric_differences(original_metrics, calibrated_metrics)
            if original_metrics is not None
            else None
        ),
        "classification_differences_calibrated_minus_uncalibrated": {
            key: calibrated_classification[key] - original_classification[key]
            for key in ("accuracy", "macro_f1")
        },
        "selection": decision,
        "known_limitations": [split_manifest["known_limitation"]],
    }
    output = Path(output_dir)
    _write_json(output / parameter_name, parameter_artifact)
    _write_json(output / comparison_name, comparison)
    return comparison


def _classification_summary_from_labels(
    predicted: Sequence[str], actual: Sequence[str]
) -> dict[str, Any]:
    return evaluate_predictions(list(actual), list(predicted), list(LABEL_ORDER))[0]


def _metric_differences(
    original: dict[str, Any], calibrated: dict[str, Any]
) -> dict[str, float]:
    return {
        key: calibrated[key] - original[key]
        for key in ("ece", "multiclass_brier_score", "negative_log_likelihood")
    }


def _calibration_decision(
    original: dict[str, Any] | None, calibrated: dict[str, Any]
) -> dict[str, Any]:
    if original is None:
        return {
            "selected": "sigmoid_calibration",
            "reason": "Linear SVM raw scores are not probabilities; calibrated probabilities are required.",
        }
    keys = ("ece", "multiclass_brier_score", "negative_log_likelihood")
    improvements = sum(calibrated[key] < original[key] for key in keys)
    selected = "calibrated" if (
        calibrated["negative_log_likelihood"] < original["negative_log_likelihood"]
        and improvements >= 2
    ) else "none"
    return {
        "selected": selected,
        "metrics_improved_count": improvements,
        "rule": "Select calibration when held-out NLL and at least one other probability metric improve.",
        "reason": (
            "Held-out probability evidence supports calibration."
            if selected == "calibrated"
            else "Held-out probability evidence does not justify replacing original probabilities."
        ),
    }


def _load_calibrated_probabilities(
    output_dir: str | Path, model: ModelName, subset: dict[str, Any]
) -> np.ndarray:
    output = Path(output_dir)
    if model == "transformer":
        scaler = TemperatureScaler.from_dict(
            _read_json(output / "temperature_scaling.json")
        )
        return scaler.transform(subset["logits_or_scores"])
    artifact = _read_json(
        output
        / (
            "logistic_regression_sigmoid_scaling.json"
            if model == "logistic_regression"
            else "linear_svm_sigmoid_calibration.json"
        )
    )
    calibrator = SigmoidScoreCalibrator.from_dict(artifact)
    scores = subset["logits_or_scores"]
    if model == "logistic_regression":
        scores = np.log(
            np.clip(subset["probabilities"], np.finfo(float).tiny, 1.0)
        )
    return calibrator.transform(scores)


def evaluate_model_thresholds(
    run_root: str | Path,
    output_dir: str | Path,
    model: ModelName,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate uncalibrated/calibrated confidence on threshold-selection only."""

    _, threshold_ids, split_manifest = _load_subset_ids(output_dir)
    normalized, selected = load_selected_predictions(run_root, model, output_dir)
    evaluation = _subset(normalized, threshold_ids)
    labels = _label_ids(evaluation["actual_labels"])
    variants: dict[str, Any] = {}
    if evaluation["probabilities"] is not None:
        variants["uncalibrated"] = evaluate_thresholds(
            evaluation["probabilities"], labels, LABEL_ORDER, thresholds=thresholds
        )
    else:
        variants["uncalibrated"] = {
            "status": "not_applicable",
            "reason": "raw Linear SVM decision scores are not probabilities",
        }
    calibrated = _load_calibrated_probabilities(output_dir, model, evaluation)
    variants["calibrated"] = evaluate_thresholds(
        calibrated, labels, LABEL_ORDER, thresholds=thresholds
    )
    return {
        "model": model,
        "model_identity": selected["models"][model],
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "selection_subset": "threshold_selection",
        "calibration_split_id": split_manifest["calibration_split_id"],
        "test_set_used": False,
        "thresholds": [float(value) for value in thresholds],
        "variants": variants,
    }


def run_threshold_analysis(
    run_root: str | Path,
    output_dir: str | Path,
    models: Sequence[ModelName] = MODEL_NAMES,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    artifact = {
        "phase": "2.10",
        "confidence_definition": "maximum_class_probability",
        "decision_rule": "accepted when confidence >= threshold; otherwise needs_review",
        "rejected_predictions_are_not_a_fifth_class": True,
        "models": {
            model: evaluate_model_thresholds(
                run_root, output_dir, model, thresholds=thresholds
            )
            for model in models
        },
    }
    _write_json(Path(output_dir) / "threshold_analysis.json", artifact)
    return artifact


def generate_final_policies(
    run_root: str | Path,
    output_dir: str | Path,
    *,
    minimum_selective_accuracy: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist calibration choices and a pending or business-constrained threshold."""

    root, output = Path(run_root), Path(output_dir)
    selected = verify_phase29_policy(root)
    files = {
        "transformer": output / "distilbert_calibration_comparison.json",
        "logistic_regression": output / "logistic_regression_calibration_comparison.json",
        "linear_svm": output / "linear_svm_calibration_comparison.json",
    }
    calibration_models: dict[str, Any] = {}
    for model, path in files.items():
        comparison = _read_json(path)
        choice = comparison["selection"]["selected"]
        if model == "transformer":
            method = "temperature_scaling" if choice == "calibrated" else "none"
        elif model == "linear_svm":
            method = "sigmoid_calibration"
        else:
            method = "sigmoid_calibration" if choice == "calibrated" else "none"
        calibration_models[model] = {
            "model_identity": selected["models"][model],
            "calibration_method": method,
            "parameter_reference": (
                comparison["parameter_artifact"] if method != "none" else None
            ),
            "validation_evidence": str(path),
            "decision_rationale": comparison["selection"]["reason"],
            "known_limitations": comparison["known_limitations"],
        }
    calibration_policy = {
        "phase": "2.10",
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "selection_data": "threshold_selection subset of frozen validation",
        "test_set_used": False,
        "models": calibration_models,
    }
    threshold_analysis = _read_json(output / "threshold_analysis.json")
    threshold_models: dict[str, Any] = {}
    for model, details in threshold_analysis["models"].items():
        method = calibration_models[model]["calibration_method"]
        variant = "calibrated" if method != "none" else "uncalibrated"
        evaluations = details["variants"][variant]
        selected_threshold = (
            select_threshold_for_accuracy(evaluations, minimum_selective_accuracy)
            if minimum_selective_accuracy is not None
            else None
        )
        threshold_models[model] = {
            "model_identity": selected["models"][model],
            "calibration_method": method,
            "confidence_definition": "maximum_class_probability",
            "status": "selected" if selected_threshold is not None else "pending",
            "selected_threshold": (
                selected_threshold["threshold"] if selected_threshold else None
            ),
            "selection_objective": (
                "maximize coverage subject to minimum selective accuracy"
                if minimum_selective_accuracy is not None
                else "pending business acceptance criterion"
            ),
            "target_selective_accuracy": minimum_selective_accuracy,
            "observed_selective_accuracy": (
                selected_threshold["selective_accuracy"] if selected_threshold else None
            ),
            "observed_coverage": (
                selected_threshold["coverage"] if selected_threshold else None
            ),
            "supporting_artifact": str(output / "threshold_analysis.json"),
            "selection_dataset": "threshold_selection subset of frozen validation",
        }
    threshold_policy = {
        "phase": "2.10",
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "test_set_used": False,
        "models": threshold_models,
        "operational_threshold_status": (
            "selected_with_caller_supplied_criterion"
            if minimum_selective_accuracy is not None
            else "pending_business_acceptance_criterion"
        ),
    }
    _write_json(output / "final_calibration_policy.json", calibration_policy)
    _write_json(output / "final_threshold_policy.json", threshold_policy)
    return calibration_policy, threshold_policy


def apply_prediction_decision(
    model: ModelName,
    model_outputs: Sequence[float],
    calibration_policy: dict[str, Any],
    threshold_policy: dict[str, Any],
) -> dict[str, Any]:
    """Apply persisted policies to one output vector without ground truth."""

    calibration = calibration_policy["models"][model]
    threshold = threshold_policy["models"][model]
    if threshold.get("status") != "selected" or threshold.get("selected_threshold") is None:
        raise ValueError("operational threshold is pending for this model")
    method = calibration["calibration_method"]
    values = np.asarray(model_outputs, dtype=np.float64).reshape(1, -1)
    if values.shape[1] != len(LABEL_ORDER) or not np.isfinite(values).all():
        raise ValueError("model outputs must contain one finite value per frozen class")
    calibrated = method != "none"
    if method == "temperature_scaling":
        scaler = TemperatureScaler.from_dict(
            _read_json(calibration["parameter_reference"])
        )
        probabilities = scaler.transform(values)[0]
    elif method == "sigmoid_calibration":
        calibrator = SigmoidScoreCalibrator.from_dict(
            _read_json(calibration["parameter_reference"])
        )
        scores = values
        if model == "logistic_regression":
            validate_probabilities(values, expected_classes=len(LABEL_ORDER))
            scores = np.log(np.clip(values, np.finfo(float).tiny, 1.0))
        probabilities = calibrator.transform(scores)[0]
    elif method == "none":
        probabilities = validate_probabilities(
            values, expected_classes=len(LABEL_ORDER)
        )[0]
    else:
        raise ValueError(f"unsupported calibration method: {method}")
    return apply_threshold(
        probabilities,
        LABEL_ORDER,
        float(threshold["selected_threshold"]),
        calibrated=calibrated,
    )


def render_final_report(output_dir: str | Path) -> str:
    output = Path(output_dir)
    audit = _read_json(output / "prediction_artifact_audit.json")
    calibration = _read_json(output / "final_calibration_policy.json")
    threshold = _read_json(output / "final_threshold_policy.json")
    lines = [
        "# Phase 2.10 Threshold Tuning and Calibration",
        "",
        f"Dataset `{audit['dataset_version']}`, split `{audit['split_version']}`.",
        "",
        "## Prediction audit",
        "",
    ]
    for model, details in audit["models"].items():
        lines.append(f"- {model}: {details['status']}")
    lines.extend(["", "## Calibration policy", ""])
    for model, details in calibration["models"].items():
        lines.append(f"- {model}: `{details['calibration_method']}`")
    lines.extend(["", "## Threshold policy", ""])
    for model, details in threshold["models"].items():
        lines.append(
            f"- {model}: {details['status']}; threshold={details['selected_threshold']}"
        )
    lines.extend(
        [
            "",
            "## Methodological limitation",
            "",
            "The frozen validation split previously influenced model/checkpoint selection. "
            "The internal calibration-fitting and threshold-selection subsets are disjoint, "
            "but they are not fully independent test data. No final test predictions are used.",
            "",
        ]
    )
    report = "\n".join(lines)
    _atomic_write_text(output / "threshold_calibration_report.md", report)
    return report

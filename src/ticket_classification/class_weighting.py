"""Phase 2.9 class-imbalance analysis and controlled weighting experiments."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ticket_classification.classical import (
    LinearSVMConfig,
    LogisticRegressionConfig,
    predict_linear_svm_records,
    predict_records,
    train_linear_svm,
    train_logistic_regression,
    validate_features,
)
from ticket_classification.classical_outputs import (
    ClassicalInputs,
    _write_json,
)
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.feature_outputs import (
    _atomic_joblib_dump,
    load_feature_config,
    load_feature_metadata,
    load_sparse_features,
    load_vectorizer,
)
from ticket_classification.split_outputs import (
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    _atomic_write_text,
    load_classification_records,
    load_split_manifest,
)
from ticket_pipeline.versioning import fingerprint_value


WeightingStrategy = Literal["none", "balanced"]


class ClassWeightPolicy(BaseModel):
    """A deterministic policy calculated exclusively from training labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    split_version: str
    weighting_strategy: WeightingStrategy
    class_order: list[str] = Field(min_length=2)
    class_counts: dict[str, int]
    class_weights: dict[str, float]
    calculation_formula: str
    source_split: Literal["train"] = "train"


def calculate_balanced_class_weights(
    labels: Sequence[str], class_order: Sequence[str]
) -> dict[str, float]:
    """Calculate N / (K * N_c) in the supplied frozen class order."""

    ordered = list(class_order)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("class_order must contain at least two unique labels")
    if not labels:
        raise ValueError("training labels must not be empty")
    unknown = sorted(set(labels) - set(ordered))
    if unknown:
        raise ValueError(f"training labels contain unknown classes: {unknown}")
    counts = Counter(labels)
    missing = [label for label in ordered if counts[label] == 0]
    if missing:
        raise ValueError(f"training labels are missing frozen classes: {missing}")
    total = len(labels)
    class_count = len(ordered)
    return {
        label: total / (class_count * counts[label])
        for label in ordered
    }


def build_class_weight_policy(inputs, strategy: WeightingStrategy = "balanced") -> ClassWeightPolicy:
    labels = [record.label for record in inputs.splits.train]
    counts = Counter(labels)
    weights = (
        calculate_balanced_class_weights(labels, inputs.classes)
        if strategy == "balanced"
        else {label: 1.0 for label in inputs.classes}
    )
    return ClassWeightPolicy(
        dataset_version=inputs.feature_metadata.dataset_version,
        split_version=inputs.feature_metadata.split_version,
        weighting_strategy=strategy,
        class_order=inputs.classes,
        class_counts={label: counts[label] for label in inputs.classes},
        class_weights=weights,
        calculation_formula="w_c = N / (K * N_c)" if strategy == "balanced" else "w_c = 1",
    )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_metrics(path: Path, *, nested: bool) -> dict[str, Any]:
    value = _read_json(path)
    metrics = value.get("validation") if nested else value
    required = {"accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "per_class"}
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise ValueError(f"artifact lacks validation metrics: {path}")
    return metrics


def _load_phase29_inputs(split_dir: str | Path, feature_dir: str | Path) -> ClassicalInputs:
    """Load and verify train/validation only; never open frozen test artifacts."""

    split_root = Path(split_dir)
    feature_root = Path(feature_dir)
    manifest = load_split_manifest(split_root / "split_manifest.json")
    train = load_classification_records(split_root / TRAIN_FILENAME)
    validation = load_classification_records(split_root / VALIDATION_FILENAME)
    groups = {"train": train, "validation": validation}
    for name, records in groups.items():
        expected_ids = getattr(manifest, f"{name}_record_ids")
        expected_count = getattr(manifest, f"{name}_count")
        expected_classes = getattr(manifest, f"class_counts_{name}")
        if len(records) != expected_count:
            raise ValueError(f"{name} count differs from frozen split manifest")
        if [record.record_id for record in records] != expected_ids:
            raise ValueError(f"{name} membership/order differs from frozen split manifest")
        if Counter(record.label for record in records) != expected_classes:
            raise ValueError(f"{name} labels differ from frozen split manifest")
    vectorizer = load_vectorizer(feature_root / "vectorizer.joblib")
    feature_config = load_feature_config(feature_root / "feature_config.json")
    metadata = load_feature_metadata(feature_root / "feature_metadata.json")
    matrices = {
        name: load_sparse_features(feature_root / f"{name}_features.npz")
        for name in groups
    }
    classes = sorted(manifest.class_counts_full)
    if metadata.dataset_version != manifest.dataset_version:
        raise ValueError("TF-IDF dataset version differs from frozen split")
    expected_split_version = "split_" + fingerprint_value(manifest.model_dump(mode="json"))[:12]
    if metadata.split_version != expected_split_version:
        raise ValueError("TF-IDF split version differs from frozen split manifest")
    if metadata.matrix_shapes["train"] != matrices["train"].shape:
        raise ValueError("train TF-IDF shape differs from metadata")
    if metadata.matrix_shapes["validation"] != matrices["validation"].shape:
        raise ValueError("validation TF-IDF shape differs from metadata")
    for name, records in groups.items():
        matrix = matrices[name]
        validate_features(matrix, [record.label for record in records], metadata.feature_count)
        transformed = vectorizer.transform([record.text for record in records]).tocsr()
        difference = transformed - matrix
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
            raise ValueError(f"{name} features differ from frozen transformations")
    splits = SimpleNamespace(train=train, validation=validation, manifest=manifest)
    return ClassicalInputs(
        splits=splits, feature_dir=feature_root, vectorizer=vectorizer,
        feature_config=feature_config, feature_metadata=metadata,
        matrices=matrices, classes=classes,
    )


def run_class_imbalance_analysis(
    split_dir: str | Path,
    feature_dir: str | Path,
    baseline_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze train-only frequency and available unweighted validation evidence."""

    inputs = _load_phase29_inputs(split_dir, feature_dir)
    policy = build_class_weight_policy(inputs)
    root = Path(baseline_root)
    model_sources = {
        "logistic_regression": (root / "logistic_regression" / "metrics.json", True),
        "linear_svm": (root / "linear_svm" / "metrics.json", True),
        "distilbert": (root / "transformer_training" / "validation_metrics.json", False),
    }
    models: dict[str, Any] = {}
    for name, (path, nested) in model_sources.items():
        if path.is_file():
            models[name] = {"status": "available", "source": str(path),
                            "validation_metrics": _validation_metrics(path, nested=nested)}
        else:
            models[name] = {"status": "missing", "source": str(path),
                            "validation_metrics": None}
    counts = policy.class_counts
    total = sum(counts.values())
    majority = max(policy.class_order, key=lambda label: counts[label])
    minority = min(policy.class_order, key=lambda label: counts[label])
    weaknesses = {
        name: [label for label in policy.class_order
               if details["validation_metrics"] is not None
               and details["validation_metrics"]["per_class"][label]["recall"]
               < details["validation_metrics"]["macro_recall"]]
        for name, details in models.items()
    }
    analysis = {
        "dataset_version": policy.dataset_version,
        "split_version": policy.split_version,
        "source_split": "train",
        "total_training_examples": total,
        "number_of_classes": len(policy.class_order),
        "class_order": policy.class_order,
        "class_counts": counts,
        "class_percentages": {label: counts[label] / total for label in policy.class_order},
        "majority_class": majority,
        "minority_class": minority,
        "majority_to_minority_ratio": counts[majority] / counts[minority],
        "unweighted_models": models,
        "classes_below_each_model_macro_recall": weaknesses,
        "weighting_experiment_warranted": True,
        "reasoning": (
            "The frozen training distribution is unequal and available validation results show "
            "class-specific trade-offs. A controlled experiment is warranted, but weighting is "
            "not presumed to improve validation Macro F1. Frequency alone does not establish causation."
        ),
        "alternative_explanations": [
            "overlapping class semantics", "label ambiguity", "insufficient examples",
            "potential data-quality problems",
        ],
    }
    analysis["analysis_id"] = "imbalance_" + fingerprint_value(analysis)[:12]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "class_weight_policy.json", policy.model_dump(mode="json"))
    _write_json(output / "class_imbalance_analysis.json", analysis)
    _atomic_write_text(output / "class_imbalance_analysis.md", render_imbalance_report(analysis))
    return analysis


def render_imbalance_report(analysis: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {label} | {analysis['class_counts'][label]} | {analysis['class_percentages'][label]:.2%} |"
        for label in analysis["class_order"]
    )
    model_rows = []
    for name, details in analysis["unweighted_models"].items():
        metrics = details["validation_metrics"]
        model_rows.append(
            f"| {name} | {details['status']} | "
            + (f"{metrics['macro_f1']:.6f} |" if metrics else "N/A |")
        )
    per_class_sections = []
    for name, details in analysis["unweighted_models"].items():
        metrics = details["validation_metrics"]
        if metrics is None:
            continue
        metric_rows = "\n".join(
            f"| {label} | {metrics['per_class'][label]['precision']:.6f} | "
            f"{metrics['per_class'][label]['recall']:.6f} | "
            f"{metrics['per_class'][label]['f1']:.6f} | "
            f"{metrics['per_class'][label]['support']} |"
            for label in analysis["class_order"]
        )
        per_class_sections.append(
            f"### {name}\n\n| Class | Precision | Recall | F1 | Support |\n"
            f"| --- | ---: | ---: | ---: | ---: |\n{metric_rows}"
        )
    return (
        "# Phase 2.9 Class Imbalance Analysis\n\n"
        f"Dataset `{analysis['dataset_version']}`, split `{analysis['split_version']}`. "
        "Weights and frequencies use the training split only.\n\n"
        "| Class | Training count | Percentage |\n| --- | ---: | ---: |\n" + rows + "\n\n"
        f"Majority: **{analysis['majority_class']}**. Minority: **{analysis['minority_class']}**. "
        f"Ratio: **{analysis['majority_to_minority_ratio']:.3f}:1**.\n\n"
        "## Existing unweighted validation evidence\n\n"
        "| Model | Status | Macro F1 |\n| --- | --- | ---: |\n" + "\n".join(model_rows) + "\n\n"
        + "\n\n".join(per_class_sections) + "\n\n"
        + "## Decision\n\n" + analysis["reasoning"] + "\n"
    )


def _metric_comparison(unweighted: dict[str, Any], weighted: dict[str, Any]) -> dict[str, Any]:
    summary_keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
    summary = {
        key: {"unweighted": unweighted[key], "weighted": weighted[key],
              "difference": weighted[key] - unweighted[key]}
        for key in summary_keys
    }
    per_class = {
        label: {
            metric: {"unweighted": unweighted["per_class"][label][metric],
                     "weighted": weighted["per_class"][label][metric],
                     "difference": weighted["per_class"][label][metric]
                     - unweighted["per_class"][label][metric]}
            for metric in ("precision", "recall", "f1")
        }
        for label in weighted["class_order"]
    }
    return {"summary": summary, "per_class": per_class}


def _confusion_comparison(unweighted: dict[str, Any], weighted: dict[str, Any]) -> dict[str, Any]:
    if unweighted["row_labels"] != weighted["row_labels"] or unweighted["column_labels"] != weighted["column_labels"]:
        raise ValueError("weighted and unweighted confusion-matrix labels differ")
    differences = [
        [weighted_value - unweighted_value for unweighted_value, weighted_value in zip(old, new)]
        for old, new in zip(unweighted["values"], weighted["values"])
    ]
    labels = weighted["row_labels"]
    changed_errors = sorted(
        (
            {"actual_label": labels[row], "predicted_label": labels[column], "difference": value}
            for row, values in enumerate(differences)
            for column, value in enumerate(values)
            if row != column and value
        ),
        key=lambda item: (-abs(item["difference"]), item["actual_label"], item["predicted_label"]),
    )
    return {
        "row_labels": labels,
        "column_labels": weighted["column_labels"],
        "values_difference_weighted_minus_unweighted": differences,
        "largest_changed_misclassifications": changed_errors[:10],
    }


def run_weighted_classical_experiment(
    model_type: Literal["logistic_regression", "linear_svm"],
    split_dir: str | Path,
    feature_dir: str | Path,
    baseline_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train one balanced model and evaluate validation only; never load test metrics for selection."""

    inputs = _load_phase29_inputs(split_dir, feature_dir)
    policy = build_class_weight_policy(inputs)
    train_labels = [record.label for record in inputs.splits.train]
    validation_records = inputs.splits.validation
    matrix = inputs.matrices["validation"]
    if model_type == "logistic_regression":
        config = LogisticRegressionConfig(class_weight="balanced")
        trained = train_logistic_regression(inputs.matrices["train"], train_labels, config)
        prediction_rows = predict_records(trained.model, matrix, validation_records, inputs.classes)
    elif model_type == "linear_svm":
        config = LinearSVMConfig(class_weight="balanced")
        trained = train_linear_svm(inputs.matrices["train"], train_labels, config)
        prediction_rows = predict_linear_svm_records(
            trained.model, matrix, validation_records, inputs.classes
        )
    else:  # pragma: no cover - Literal protects typed callers; CLI validates strings
        raise ValueError(f"unsupported model_type: {model_type}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "weighted_model.joblib"
    _atomic_joblib_dump(model_path, trained.model)
    reloaded = joblib.load(model_path)
    if list(reloaded.predict(matrix)) != [row["predicted_label"] for row in prediction_rows]:
        raise RuntimeError("reloaded weighted model predictions differ")
    metrics, confusion = evaluate_predictions(
        [row["actual_label"] for row in prediction_rows],
        [row["predicted_label"] for row in prediction_rows], inputs.classes,
    )
    baseline_path = Path(baseline_dir) / "metrics.json"
    baseline = _validation_metrics(baseline_path, nested=True)
    comparison = _metric_comparison(baseline, metrics)
    baseline_confusion = _read_json(Path(baseline_dir) / "confusion_matrix.json")["validation"]
    comparison["confusion_matrix"] = _confusion_comparison(baseline_confusion, confusion)
    comparison.update({
        "model_type": model_type, "selection_split": "validation",
        "test_set_used": False,
        "minority_class": min(policy.class_order, key=lambda label: policy.class_counts[label]),
        "decision_rule": "Prefer weighted only when validation Macro F1 improves; inspect all per-class trade-offs.",
    })
    manifest = {
        "experiment_id": f"weighted_{model_type}_" + fingerprint_value({
            "policy": policy.model_dump(mode="json"), "config": config.model_dump(mode="json")
        })[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": policy.dataset_version,
        "split_version": policy.split_version,
        "model_type": model_type,
        "configuration": config.model_dump(mode="json"),
        "weighting_policy": policy.model_dump(mode="json"),
        "training_time": trained.training_time,
        "validation_metrics": metrics,
        "test_set_used": False,
        "reload_verified": True,
    }
    _write_json(output / "config.json", config.model_dump(mode="json"))
    _write_json(output / "validation_metrics.json", metrics)
    _write_json(output / "confusion_matrix.json", confusion)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "experiment_manifest.json", manifest)
    _atomic_write_text(
        output / "validation_predictions.jsonl",
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in prediction_rows),
    )
    return manifest


def run_weighting_comparison(
    baseline_root: str | Path, weighting_root: str | Path
) -> dict[str, Any]:
    """Compare available validation results and leave missing experiments pending."""

    baseline = Path(baseline_root)
    output = Path(weighting_root)
    sources = {
        "logistic_regression": {
            "unweighted_metrics": baseline / "logistic_regression" / "metrics.json",
            "weighted_metrics": output / "logistic_regression" / "validation_metrics.json",
            "unweighted_nested": True,
            "weighted_nested": False,
            "unweighted_confusion": baseline / "logistic_regression" / "confusion_matrix.json",
            "weighted_confusion": output / "logistic_regression" / "confusion_matrix.json",
            "confusion_nested": True,
        },
        "linear_svm": {
            "unweighted_metrics": baseline / "linear_svm" / "metrics.json",
            "weighted_metrics": output / "linear_svm" / "validation_metrics.json",
            "unweighted_nested": True,
            "weighted_nested": False,
            "unweighted_confusion": baseline / "linear_svm" / "confusion_matrix.json",
            "weighted_confusion": output / "linear_svm" / "confusion_matrix.json",
            "confusion_nested": True,
        },
        "distilbert": {
            "unweighted_metrics": baseline / "transformer_training" / "validation_metrics.json",
            "weighted_metrics": output / "transformer" / "validation_metrics.json",
            "unweighted_nested": False,
            "weighted_nested": False,
            "unweighted_confusion": baseline / "transformer_training" / "validation_confusion_matrix.json",
            "weighted_confusion": output / "transformer" / "validation_confusion_matrix.json",
            "confusion_nested": False,
            "weighted_manifest": output / "transformer" / "training_manifest.json",
        },
    }
    comparisons: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for name, source in sources.items():
        unweighted_path = source["unweighted_metrics"]
        weighted_path = source["weighted_metrics"]
        transformer_ready = True
        transformer_reason = None
        if name == "distilbert":
            manifest_path = source["weighted_manifest"]
            if not manifest_path.is_file():
                transformer_ready = False
                transformer_reason = "Full weighted training manifest is missing."
            else:
                manifest = _read_json(manifest_path)
                if manifest.get("smoke_test") is not False:
                    transformer_ready = False
                    transformer_reason = "Smoke-test results cannot select the weighting policy."
                elif manifest.get("class_weighting") != "balanced":
                    transformer_ready = False
                    transformer_reason = "Training manifest does not identify balanced weighting."
                elif manifest.get("test_set_loaded") is not False:
                    transformer_ready = False
                    transformer_reason = "Training manifest does not preserve test isolation."
        if not unweighted_path.is_file() or not weighted_path.is_file() or not transformer_ready:
            comparisons[name] = {
                "status": "PENDING", "unweighted_source": str(unweighted_path),
                "weighted_source": str(weighted_path), "comparison": None,
            }
            decisions[name] = {
                "status": "PENDING", "selected_weighting_policy": None,
                "reasoning": transformer_reason or "Both unweighted and weighted validation results are required.",
            }
            continue
        unweighted = _validation_metrics(unweighted_path, nested=source["unweighted_nested"])
        weighted = _validation_metrics(weighted_path, nested=source["weighted_nested"])
        comparison = _metric_comparison(unweighted, weighted)
        if source["unweighted_confusion"].is_file() and source["weighted_confusion"].is_file():
            unweighted_confusion = _read_json(source["unweighted_confusion"])
            if source["confusion_nested"]:
                unweighted_confusion = unweighted_confusion["validation"]
            comparison["confusion_matrix"] = _confusion_comparison(
                unweighted_confusion, _read_json(source["weighted_confusion"])
            )
        selected = "balanced" if weighted["macro_f1"] > unweighted["macro_f1"] else "none"
        comparisons[name] = {"status": "COMPLETE", "comparison": comparison}
        decisions[name] = {
            "status": "SELECTED", "selected_weighting_policy": selected,
            "validation_macro_f1": max(weighted["macro_f1"], unweighted["macro_f1"]),
            "validation_macro_f1_difference": weighted["macro_f1"] - unweighted["macro_f1"],
            "per_class_changes": comparison["per_class"],
            "reasoning": (
                "Balanced weighting improved validation Macro F1; inspect per-class deltas before final use."
                if selected == "balanced"
                else "Balanced weighting did not improve validation Macro F1; retain the simpler unweighted policy."
            ),
            "known_limitations": ["Validation evidence only", "Loss values across weighting policies are not directly comparable"],
        }
    report = {
        "selection_split": "validation", "test_set_used": False,
        "models": comparisons, "final_weighting_policy": decisions,
        "phase_complete": all(value["status"] == "SELECTED" for value in decisions.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "class_weighting_comparison.json", report)
    _write_json(output / "final_weighting_policy.json", decisions)
    _atomic_write_text(output / "class_weighting_comparison.md", render_weighting_comparison(report))
    return report


def render_weighting_comparison(report: dict[str, Any]) -> str:
    lines = ["# Phase 2.9 Weighted vs Unweighted Comparison", "",
             "Selection uses the frozen validation split only. The test set is not used.", ""]
    for name, details in report["models"].items():
        lines.extend([f"## {name}", ""])
        if details["status"] == "PENDING":
            lines.extend(["**PENDING** — both validation artifacts are required.", ""])
            continue
        lines.extend(["| Metric | Unweighted | Weighted | Difference |",
                      "| --- | ---: | ---: | ---: |"])
        for metric, row in details["comparison"]["summary"].items():
            lines.append(f"| {metric} | {row['unweighted']:.6f} | {row['weighted']:.6f} | {row['difference']:+.6f} |")
        lines.append("")
        lines.extend(["| Class | Metric | Unweighted | Weighted | Difference |",
                      "| --- | --- | ---: | ---: | ---: |"])
        for label, metrics in details["comparison"]["per_class"].items():
            for metric, row in metrics.items():
                lines.append(
                    f"| {label} | {metric} | {row['unweighted']:.6f} | "
                    f"{row['weighted']:.6f} | {row['difference']:+.6f} |"
                )
        changed = details["comparison"].get("confusion_matrix", {}).get(
            "largest_changed_misclassifications", []
        )
        if changed:
            lines.extend(["", "Largest changed off-diagonal confusion counts (weighted minus unweighted):"])
            lines.extend(
                f"- {row['actual_label']} → {row['predicted_label']}: {row['difference']:+d}"
                for row in changed
            )
        lines.append("")
    lines.extend(["## Current policy", "", "```json",
                  json.dumps(report["final_weighting_policy"], sort_keys=True, indent=2), "```", ""])
    return "\n".join(lines)

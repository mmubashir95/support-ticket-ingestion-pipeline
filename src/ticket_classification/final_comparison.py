"""Phase 2.12 final, test-only comparison of the three selected classifiers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest

from ticket_classification.benchmarking import DEFAULT_RUN_ROOT, artifact_sizes, resolve_artifact_path
from ticket_classification.calibration_workflow import verify_phase29_policy
from ticket_classification.classical_outputs import load_linear_svm, load_logistic_regression
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.feature_outputs import load_sparse_features
from ticket_classification.features import load_frozen_splits
from ticket_classification.split_outputs import _atomic_write_text


MODEL_ORDER = ("logistic_regression", "linear_svm", "transformer")
DISPLAY_NAMES = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "transformer": "Transformer classifier",
}
EXPLAINABILITY = {"logistic_regression": "High", "linear_svm": "High", "transformer": "Medium/Low"}
PROBABILITY_OUTPUT = {
    "logistic_regression": "Native; temperature-scaled",
    "linear_svm": "Calibrated from margins",
    "transformer": "Native softmax; temperature-scaled",
}
OPERATIONAL_COMPLEXITY = {"logistic_regression": "Low", "linear_svm": "Low", "transformer": "Higher"}
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "final_comparison"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required Phase 2.12 input is missing: {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required Phase 2.12 input is missing: {path}") from error
    if not rows:
        raise ValueError(f"prediction artifact is empty: {path}")
    return rows


def align_predictions(
    record_ids: Sequence[str], actual: Sequence[str], rows: Sequence[dict[str, Any]], model: str
) -> list[str]:
    """Strictly align a prediction artifact to the frozen test population."""
    if len(record_ids) != len(actual) or len(set(record_ids)) != len(record_ids):
        raise ValueError("frozen test identities must be aligned and unique")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in indexed:
            raise ValueError(f"{model} predictions contain a missing or duplicate record_id")
        indexed[record_id] = row
    if set(indexed) != set(record_ids):
        raise ValueError(f"{model} predictions do not exactly match the frozen test membership")
    predictions = []
    for record_id, true_label in zip(record_ids, actual):
        row = indexed[record_id]
        row_actual = row.get("actual_label", row.get("true_label"))
        if row_actual != true_label:
            raise ValueError(f"{model} true label differs from the frozen test record")
        predicted = row.get("predicted_label")
        if not isinstance(predicted, str):
            raise ValueError(f"{model} prediction has no predicted_label")
        predictions.append(predicted)
    return predictions


def paired_significance(actual: Sequence[str], left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    """Exact paired correctness comparison (two-sided McNemar/binomial test)."""
    if not actual or len(actual) != len(left) or len(actual) != len(right):
        raise ValueError("paired comparison requires aligned nonempty predictions")
    left_only = sum(a == l and a != r for a, l, r in zip(actual, left, right))
    right_only = sum(a != l and a == r for a, l, r in zip(actual, left, right))
    discordant = left_only + right_only
    p_value = float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_count": discordant,
        "two_sided_exact_p_value": p_value,
        "statistically_significant_at_0_05": p_value < 0.05,
        "test": "two-sided exact McNemar (binomial form)",
    }


def paired_bootstrap_macro_f1(
    actual: Sequence[str], left: Sequence[str], right: Sequence[str], labels: Sequence[str],
    *, repetitions: int = 2000, seed: int = 42,
) -> dict[str, float]:
    """Estimate a paired 95% interval for left minus right Macro F1."""
    if repetitions < 1 or not actual or len(actual) != len(left) or len(actual) != len(right):
        raise ValueError("invalid paired bootstrap inputs")
    rng = np.random.default_rng(seed)
    actual_array, left_array, right_array = map(np.asarray, (actual, left, right))
    differences = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.integers(0, len(actual_array), len(actual_array))
        left_metrics, _ = evaluate_predictions(actual_array[sample].tolist(), left_array[sample].tolist(), labels)
        right_metrics, _ = evaluate_predictions(actual_array[sample].tolist(), right_array[sample].tolist(), labels)
        differences[index] = left_metrics["macro_f1"] - right_metrics["macro_f1"]
    return {
        "left_minus_right_mean": float(differences.mean()),
        "ci95_lower": float(np.percentile(differences, 2.5)),
        "ci95_upper": float(np.percentile(differences, 97.5)),
        "repetitions": repetitions,
        "seed": seed,
    }


def _benchmark_summary(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    summary = {name: {"cpu_p50_ms": None, "cpu_p95_ms": None, "gpu_p50_ms": None,
                      "model_size_bytes": None, "cost_per_1000": None} for name in MODEL_ORDER}
    if not data:
        return summary
    if data.get("sample", {}).get("split") != "test":
        raise ValueError("Phase 2.11 benchmark must use the frozen test split")
    for result in data.get("results", []):
        model, device = result.get("model"), result.get("device")
        if model not in summary:
            continue
        summary[model]["model_size_bytes"] = result.get("artifact_size", {}).get("total_bytes")
        batch_one = next((row for row in result.get("measurements", []) if row.get("batch_size") == 1), None)
        if not batch_one:
            continue
        prefix = "cpu" if device == "cpu" else "gpu"
        summary[model][f"{prefix}_p50_ms"] = batch_one["p50_batch_seconds"] * 1000
        if device == "cpu":
            summary[model]["cpu_p95_ms"] = batch_one["p95_batch_seconds"] * 1000
        cost = batch_one.get("cost")
        if cost:
            summary[model]["cost_per_1000"] = cost.get("cost_per_1000_tickets")
    return summary


def _selected_test_predictions(run_root: Path, selected: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    splits = load_frozen_splits(run_root / "splits")
    records = splits.test
    actual = [record.label for record in records]
    ids = [record.record_id for record in records]
    features = load_sparse_features(run_root / "tfidf" / "test_features.npz")
    predictions: dict[str, list[str]] = {}
    for model in ("logistic_regression", "linear_svm"):
        path = resolve_artifact_path(selected["models"][model]["model_artifact"], run_root)
        estimator = load_logistic_regression(path) if model == "logistic_regression" else load_linear_svm(path)
        if list(estimator.classes_) != selected["label_order"]:
            raise ValueError(f"{model} class order differs from the frozen policy")
        predictions[model] = estimator.predict(features).tolist()
    transformer_rows = _read_jsonl(run_root / "transformer_training" / "test_predictions.jsonl")
    predictions["transformer"] = align_predictions(ids, actual, transformer_rows, "transformer")
    return actual, predictions


def _recommend(models: dict[str, Any]) -> dict[str, Any]:
    winner = max(MODEL_ORDER, key=lambda name: models[name]["quality"]["macro_f1"])
    return {
        "selected_model": winner,
        "display_name": DISPLAY_NAMES[winner],
        "decision_rule": "highest frozen-test Macro F1; prefer the simpler model when quality is effectively tied",
        "rationale": (
            f"{DISPLAY_NAMES[winner]} has the highest final-test Macro F1. "
            "Its per-class results, calibration evidence, and measured operational profile are reported together; "
            "missing benchmark cells are not estimated."
        ),
    }


def build_final_comparison(run_root: Path = DEFAULT_RUN_ROOT, *, bootstrap_repetitions: int = 2000) -> dict[str, Any]:
    selected = verify_phase29_policy(run_root)
    actual, predictions = _selected_test_predictions(run_root, selected)
    labels = selected["label_order"]
    calibration_policy = _read_json(run_root / "threshold_calibration" / "final_calibration_policy.json")
    threshold_policy = _read_json(run_root / "threshold_calibration" / "final_threshold_policy.json")
    benchmark_path = run_root / "inference_benchmark" / "benchmark_results.json"
    benchmark = _read_json(benchmark_path) if benchmark_path.exists() else None
    engineering = _benchmark_summary(benchmark)
    models: dict[str, Any] = {}
    for model in MODEL_ORDER:
        metrics, confusion = evaluate_predictions(actual, predictions[model], labels)
        calibration = _read_json(run_root / "threshold_calibration" /
                                 ("distilbert_calibration_comparison.json" if model == "transformer" else f"{model}_calibration_comparison.json"))
        manifest_path = resolve_artifact_path(selected["models"][model]["manifest"], run_root)
        manifest = _read_json(manifest_path)
        artifact_ref = selected["models"][model].get("checkpoint") or selected["models"][model].get("model_artifact")
        artifact_path = resolve_artifact_path(artifact_ref, run_root)
        if engineering[model]["model_size_bytes"] is None:
            engineering[model]["model_size_bytes"] = artifact_sizes((artifact_path,))["total_bytes"]
        models[model] = {
            "display_name": DISPLAY_NAMES[model], "quality": metrics, "confusion_matrix": confusion,
            "training_time_seconds": manifest.get("training_seconds", manifest.get("training_time")),
            "engineering": engineering[model], "explainability": EXPLAINABILITY[model],
            "probability_output": PROBABILITY_OUTPUT[model],
            "calibration": {"method": calibration_policy["models"][model]["calibration_method"],
                            "validation_threshold_selection_ece": calibration["calibrated"]["probability_metrics"]["ece"]},
            "operational_complexity": OPERATIONAL_COMPLEXITY[model],
            "cost": "Local model; infrastructure rate not supplied" if engineering[model]["cost_per_1000"] is None else engineering[model]["cost_per_1000"],
            "selected_artifact": str(artifact_path),
        }
    pairwise = {}
    for left_index, left in enumerate(MODEL_ORDER):
        for right in MODEL_ORDER[left_index + 1:]:
            key = f"{left}_vs_{right}"
            pairwise[key] = {
                "left": left, "right": right,
                "mcnemar": paired_significance(actual, predictions[left], predictions[right]),
                "macro_f1_bootstrap": paired_bootstrap_macro_f1(
                    actual, predictions[left], predictions[right], labels,
                    repetitions=bootstrap_repetitions,
                ),
                "per_class_f1_difference": {
                    label: models[left]["quality"]["per_class"][label]["f1"] - models[right]["quality"]["per_class"][label]["f1"]
                    for label in labels
                },
            }
    missing = [f"{name} CPU P50/P95" for name in MODEL_ORDER if engineering[name]["cpu_p50_ms"] is None]
    return {
        "schema_version": "phase_2_12_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": selected["dataset_version"], "split_version": selected["split_version"],
        "evaluation_split": "test", "test_record_count": len(actual), "label_order": labels,
        "selected_model_policy": selected["models"], "models": models, "pairwise_evidence": pairwise,
        "recommendation": _recommend(models),
        "threshold_status": threshold_policy["operational_threshold_status"],
        "benchmark_evidence_status": "complete" if not missing else "partial",
        "missing_benchmark_evidence": missing,
        "limitations": [
            "The test split is used here for the final comparison and must not be used for further tuning.",
            "Calibration ECE is validation threshold-selection evidence, not a test-set estimate.",
            "Exact McNemar tests compare accuracy/correctness; paired bootstrap intervals compare Macro F1.",
            "No operational rejection threshold is selected because no business acceptance criterion was supplied.",
            "Infrastructure cost is unavailable unless an hourly rate was supplied to Phase 2.11.",
        ],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_final_report(data: dict[str, Any]) -> str:
    models = data["models"]
    headers = [DISPLAY_NAMES[name] for name in MODEL_ORDER]
    lines = ["# Phase 2.12 Final Three-Model Comparison", "", "## Final recommendation", "",
             f"**Selected model: {data['recommendation']['display_name']}.** {data['recommendation']['rationale']}", "",
             "## Fair-comparison contract", "",
             f"All quality rows use the same frozen `{data['evaluation_split']}` split ({data['test_record_count']} records), label order, selected Phase 2.9 model artifacts, and unchanged argmax predictions. The test set is now consumed and Phase 2 quality selection is frozen.", "",
             "## Final comparison table", "", f"| Metric | {headers[0]} | {headers[1]} | {headers[2]} |", "|---|---:|---:|---:|"]
    rows = [
        ("Macro F1", lambda m: _fmt(m["quality"]["macro_f1"])),
        ("Macro Precision", lambda m: _fmt(m["quality"]["macro_precision"])),
        ("Macro Recall", lambda m: _fmt(m["quality"]["macro_recall"])),
        ("Weighted F1", lambda m: _fmt(m["quality"]["weighted_f1"])),
        ("Training time (s)", lambda m: _fmt(m["training_time_seconds"], 3)),
        ("CPU P50 latency (ms, batch 1)", lambda m: _fmt(m["engineering"]["cpu_p50_ms"], 3)),
        ("CPU P95 latency (ms, batch 1)", lambda m: _fmt(m["engineering"]["cpu_p95_ms"], 3)),
        ("Accelerator P50 (ms, batch 1)", lambda m: _fmt(m["engineering"]["gpu_p50_ms"], 3)),
        ("Model size (bytes)", lambda m: _fmt(m["engineering"]["model_size_bytes"])),
        ("Explainability", lambda m: m["explainability"]),
        ("Probability output", lambda m: m["probability_output"]),
        ("Calibrated ECE (validation evidence)", lambda m: _fmt(m["calibration"]["validation_threshold_selection_ece"])),
        ("Operational complexity", lambda m: m["operational_complexity"]),
        ("Cost per 1K", lambda m: _fmt(m["engineering"]["cost_per_1000"])),
    ]
    for title, getter in rows:
        lines.append(f"| {title} | " + " | ".join(getter(models[name]) for name in MODEL_ORDER) + " |")
    lines.extend(["", "## Per-class quality", "", f"| Class | " + " | ".join(headers) + " |", "|---|---:|---:|---:|"])
    for label in data["label_order"]:
        lines.append(f"| {label} F1 | " + " | ".join(_fmt(models[name]["quality"]["per_class"][label]["f1"]) for name in MODEL_ORDER) + " |")
    lines.extend(["", "## Statistical and operational meaning", ""])
    for evidence in data["pairwise_evidence"].values():
        boot, mc = evidence["macro_f1_bootstrap"], evidence["mcnemar"]
        lines.append(f"- **{DISPLAY_NAMES[evidence['left']]} vs {DISPLAY_NAMES[evidence['right']]}:** paired Macro-F1 difference 95% CI `{boot['ci95_lower']:.4f}` to `{boot['ci95_upper']:.4f}`; exact McNemar p=`{mc['two_sided_exact_p_value']:.4g}`.")
    lines.extend(["", "## Benchmark completeness", ""])
    if data["missing_benchmark_evidence"]:
        lines.append("The Phase 2.11 archive is partial. Missing measurements are shown as N/A, never inferred:")
        lines.extend(f"- {item}" for item in data["missing_benchmark_evidence"])
    else:
        lines.append("The Phase 2.11 archive contains CPU single-example evidence for all models.")
    lines.extend(["", "## Decision questions", "",
                  "The recommendation considers Macro F1 and per-class behavior first, then latency, footprint, explainability, probability handling, deployment requirements, and cost availability. A Transformer is not preferred merely because it is newer.", "",
                  "## Threshold and calibration status", "",
                  f"Calibration is included in confidence reporting, while argmax classes remain unchanged. Operational threshold status: `{data['threshold_status']}`.", "",
                  "## Limitations", ""])
    lines.extend(f"- {item}" for item in data["limitations"])
    lines.extend(["", "## Phase 2 freeze", "",
                  "The selected artifacts, frozen split identities, metrics, and recommendation in this report form the Phase 2 handoff. Any later model change requires a new versioned evaluation rather than tuning against this test report.", ""])
    return "\n".join(lines)


def write_final_comparison(data: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase2_final_model_comparison.json"
    report_path = output_dir / "phase2_final_model_comparison.md"
    _atomic_write_text(json_path, json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _atomic_write_text(report_path, render_final_report(data))
    return json_path, report_path

"""Phase 2.6: deterministic inspection of frozen baselines; never fits models."""

import json
from pathlib import Path

import numpy as np

from ticket_classification.audit import _percentile, _word_count
from ticket_classification.classical import coefficient_feature_mapping, linear_svm_decision_scores
from ticket_classification.classical_outputs import (
    _load_validated_classical_inputs, _write_json, load_linear_svm, load_logistic_regression,
)
from ticket_classification.dataset import DEFAULT_SHORT_TEXT_MIN_WORDS
from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.features import build_classification_text
from ticket_classification.split_outputs import _atomic_write_text
from ticket_pipeline.versioning import fingerprint_value

MODELS = ("logistic_regression", "linear_svm")
METRICS = ("macro_precision", "macro_recall", "macro_f1", "weighted_f1", "accuracy")


def validate_compatibility(manifests, metadata, configuration, classes):
    """Check both experiments against the authoritative frozen inputs.

    LR predates explicit label_mapping: its explicit class_order defines the
    identical mapping. Input policy is part of feature_configuration.
    """
    expected = {key: metadata[key] for key in
                ("dataset_version", "split_version", "feature_version")}
    expected.update(feature_configuration=configuration, class_order=classes,
                    target="ticket_type", task="single_label_multiclass")
    mapping = {label: index for index, label in enumerate(classes)}
    for name in MODELS:
        manifest = manifests[name]
        if manifest["model_type"] != name:
            raise ValueError(f"{name}: model_type mismatch")
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"{name}: incompatible {key}")
        if manifest.get("label_mapping", mapping) != mapping:
            raise ValueError(f"{name}: incompatible label_mapping")
        if manifest["configuration"].get("class_weight") is not None:
            raise ValueError(f"{name}: expected unweighted frozen baseline")


def index_predictions(rows):
    result = {row["record_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate prediction record_id")
    return result


def compare_per_class(results, classes):
    if any(result["class_order"] != classes or set(result["per_class"]) != set(classes)
           for result in results.values()):
        raise ValueError("incompatible metric class ordering")
    rows = []
    for label in classes:
        lr, svm = (results[name]["per_class"][label] for name in MODELS)
        if lr["support"] != svm["support"]:
            raise ValueError("incompatible class support")
        rows.append({"class": label, **{name: results[name]["per_class"][label] for name in MODELS},
                     "svm_minus_lr": {key: svm[key] - lr[key] for key in ("precision", "recall", "f1")}})
    return rows


def confusion_pairs(matrix):
    classes = matrix["row_labels"]
    if matrix["column_labels"] != classes:
        raise ValueError("confusion matrix class ordering differs")
    values = np.asarray(matrix["values"])
    if values.shape != (len(classes), len(classes)) or (values < 0).any():
        raise ValueError("invalid confusion matrix")
    pairs = [{"actual_class": actual, "predicted_class": predicted,
              "error_count": int(values[i, j]),
              "error_rate": float(values[i, j] / values[i].sum())}
             for i, actual in enumerate(classes) for j, predicted in enumerate(classes)
             if i != j and values[i, j] > 0]
    return sorted(pairs, key=lambda row: (-row["error_count"], row["actual_class"], row["predicted_class"]))


def extract_errors(rows, records, model_name):
    """Each error belongs to one actual-class FN and one predicted-class FP."""
    texts = {record.record_id: build_classification_text(record) for record in records}
    return [{**row, "true_label": row["actual_label"], "model_name": model_name,
             "text": texts[row["record_id"]], "false_negative_class": row["actual_label"],
             "false_positive_class": row["predicted_label"], "issue_category": "model_error"}
            for row in sorted(rows, key=lambda row: row["record_id"])
            if row["actual_label"] != row["predicted_label"]]


def class_errors(errors, label, kind):
    if kind not in ("false_positive", "false_negative"):
        raise ValueError("kind must be false_positive or false_negative")
    return [row for row in errors if row[f"{kind}_class"] == label]


def extract_disagreements(lr_rows, svm_rows):
    lr, svm = index_predictions(lr_rows), index_predictions(svm_rows)
    if lr.keys() != svm.keys():
        raise ValueError("prediction record identities differ")
    result = []
    for record_id in sorted(lr):
        a, b = lr[record_id], svm[record_id]
        if a["actual_label"] != b["actual_label"]:
            raise ValueError("prediction actual labels differ")
        if a["predicted_label"] == b["predicted_label"]:
            continue
        category = ("lr_correct_svm_wrong" if a["predicted_label"] == a["actual_label"] else
                    "svm_correct_lr_wrong" if b["predicted_label"] == b["actual_label"] else
                    "both_wrong_different")
        result.append({"record_id": record_id, "true_label": a["actual_label"],
                       "category": category, "logistic_regression": a, "linear_svm": b})
    return result


def minority_analysis(counts, per_class, errors):
    smallest = min(counts.values())
    labels = sorted(label for label, count in counts.items() if count == smallest)
    return [{"class": label, "full_dataset_count": counts[label],
             **{name: {**next(row[name] for row in per_class if row["class"] == label),
                       "false_negative_count": len(class_errors(errors[name], label, "false_negative")),
                       "false_positive_count": len(class_errors(errors[name], label, "false_positive"))}
                for name in MODELS}} for label in labels]


def top_features(model, vectorizer, top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    mapping = coefficient_feature_mapping(model, vectorizer)
    return {label: {
        "positive": [{"feature": feature, "weight": weight} for feature, weight in
                     sorted(weights.items(), key=lambda item: (-item[1], item[0])) if weight > 0][:top_k],
        "negative": [{"feature": feature, "weight": weight} for feature, weight in
                     sorted(weights.items(), key=lambda item: (item[1], item[0])) if weight < 0][:top_k],
    } for label, weights in mapping.items()}


def feature_overlap(features, classes):
    result = {}
    for label in classes:
        result[label] = {}
        for direction in ("positive", "negative"):
            lr, svm = ({row["feature"] for row in features[name][label][direction]} for name in MODELS)
            result[label][direction] = {"shared": sorted(lr & svm), "lr_only": sorted(lr - svm),
                                        "svm_only": sorted(svm - lr)}
    return result


def review_candidates(lr_rows, svm_rows, texts, short_threshold):
    """Heuristics for manual review only; no determination or label changes."""
    lr, svm = index_predictions(lr_rows), index_predictions(svm_rows)
    if lr.keys() != svm.keys():
        raise ValueError("prediction record identities differ")
    ambiguous, label_issues = [], []
    for record_id in sorted(lr):
        a, b = lr[record_id], svm[record_id]
        actual = a["actual_label"]
        if actual != b["actual_label"]:
            raise ValueError("prediction actual labels differ")
        both_wrong = a["predicted_label"] != actual and b["predicted_label"] != actual
        indicators = []
        if both_wrong:
            indicators.append("both_models_wrong")
        if a["predicted_label"] != b["predicted_label"]:
            indicators.append("models_disagree")
        if _word_count(texts[record_id]) < short_threshold:
            indicators.append("limited_text_context")
        example = {"record_id": record_id, "text": texts[record_id], "true_label": actual,
                   "logistic_regression": a, "linear_svm": b, "heuristic_only": True,
                   "requires_manual_review": True}
        if indicators:
            ambiguous.append({**example, "review_reason": "possible_ambiguity", "indicators": indicators})
        if both_wrong and a["predicted_label"] == b["predicted_label"]:
            label_issues.append({**example, "review_reason": "possible_label_issue",
                                 "indicators": ["both_models_predict_same_alternative"],
                                 "alternative_label": a["predicted_label"]})
    return ambiguous, label_issues


def length_analysis(records, predictions, short_cutoff, long_cutoff):
    lengths = {record.record_id: len(build_classification_text(record)) for record in records}
    groups = {"all": set(lengths), "short": {rid for rid, n in lengths.items() if n <= short_cutoff},
              "long": {rid for rid, n in lengths.items() if n >= long_cutoff}}
    result = {}
    for group, ids in groups.items():
        result[group] = {"record_count": len(ids)}
        for name in MODELS:
            wrong = sum(row["record_id"] in ids and row["actual_label"] != row["predicted_label"]
                        for row in predictions[name])
            result[group][name] = {"error_count": wrong, "error_rate": wrong / len(ids) if ids else None}
    for group in ("short", "long"):
        for name in MODELS:
            rate = result[group][name]["error_rate"]
            result[group][name]["elevated_vs_all"] = rate > result["all"][name]["error_rate"] if rate is not None else None
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path, rows):
    _atomic_write_text(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                               allow_nan=False, separators=(",", ":")) + "\n" for row in rows))


def _validate_predictions(rows, records, model, matrix, classes, name):
    indexed = index_predictions(rows)
    if set(indexed) != {record.record_id for record in records}:
        raise ValueError(f"{name}: prediction record identities differ from frozen split")
    ordered = [indexed[record.record_id] for record in records]
    predictions = model.predict(matrix)
    scores = model.predict_proba(matrix) if name == MODELS[0] else linear_svm_decision_scores(model, matrix)
    for record, row, prediction, score in zip(records, ordered, predictions, scores):
        if row["actual_label"] != record.label or row["predicted_label"] != prediction:
            raise ValueError(f"{name}: predictions differ from frozen labels/model")
        key = "probability_class_order" if name == MODELS[0] else "decision_score_class_order"
        if row[key] != classes:
            raise ValueError(f"{name}: score class ordering differs")
        persisted = row["probabilities"] if name == MODELS[0] else [row["decision_scores"][label] for label in classes]
        if name == MODELS[1] and (row.get("scores_are_probabilities") is not False or
                                row.get("score_type") != "raw_linear_svm_decision_function"):
            raise ValueError("SVM scores must be raw decision scores, not probabilities")
        if np.asarray(persisted).shape != score.shape or not np.allclose(persisted, score, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name}: saved score differs from frozen model")
    return ordered


def run_classical_model_comparison(split_dir, feature_dir, lr_dir, svm_dir, output_dir, *, top_k=20, audit_path=None):
    """Consume Phase 2.4/2.5 outputs, verify integrity, then persist analysis.

    Validation is the selection split; test is final descriptive evaluation.
    No fitting, optimization, re-timing, calibration or preprocessing changes.
    """
    output = Path(output_dir).resolve()
    sources = [Path(path).resolve() for path in (split_dir, feature_dir, lr_dir, svm_dir)]
    if any(output == source or output in source.parents or source in output.parents for source in sources):
        raise ValueError("comparison output must be separate from source artifact directories")
    inputs = _load_validated_classical_inputs(split_dir, feature_dir, phase="Phase 2.6")
    classes = inputs.classes
    directories = dict(zip(MODELS, (Path(lr_dir), Path(svm_dir))))
    manifests = {name: _read_json(path / "experiment_manifest.json") for name, path in directories.items()}
    validate_compatibility(manifests, inputs.feature_metadata.model_dump(mode="json"),
                           inputs.feature_config.model_dump(mode="json"), classes)
    models = {MODELS[0]: load_logistic_regression(Path(lr_dir) / "model.joblib"),
              MODELS[1]: load_linear_svm(Path(svm_dir) / "model.joblib")}
    saved_metrics = {name: _read_json(path / "metrics.json") for name, path in directories.items()}
    saved_confusion = {name: _read_json(path / "confusion_matrix.json") for name, path in directories.items()}
    for name, model in models.items():
        if list(model.classes_) != classes or model.n_features_in_ != inputs.feature_metadata.feature_count:
            raise ValueError(f"{name}: fitted label mapping/feature dimensions differ")
        if model.get_params()["class_weight"] is not None:
            raise ValueError(f"{name}: fitted model must be unweighted")
        if model.get_params()["C"] != manifests[name]["configuration"]["C"]:
            raise ValueError(f"{name}: fitted configuration differs")
        for key in ("training_time", "model_size_bytes", "inference_latency"):
            if saved_metrics[name][key] != manifests[name][key]:
                raise ValueError(f"{name}: inconsistent {key}")
        if (directories[name] / "model.joblib").stat().st_size != manifests[name]["model_size_bytes"]:
            raise ValueError(f"{name}: serialized size differs")
    all_records = inputs.splits.train + inputs.splits.validation + inputs.splits.test
    lengths = sorted(len(build_classification_text(record)) for record in all_records)
    short_cutoff, long_cutoff = _percentile(lengths, 5), _percentile(lengths, 95)
    short_words = DEFAULT_SHORT_TEXT_MIN_WORDS
    counts = inputs.splits.manifest.class_counts_full
    if audit_path is not None:
        audit = _read_json(audit_path)
        if audit["dataset_summary"]["dataset_version"] != inputs.feature_metadata.dataset_version:
            raise ValueError("audit dataset version differs")
        if {row["class_name"]: row["count"] for row in audit["class_distribution"]["classes"]} != counts:
            raise ValueError("audit class distribution differs")
        if audit["text_length_summary"]["character_length"]["p95"] != long_cutoff:
            raise ValueError("audit text length distribution differs")
        long_cutoff = audit["text_length_summary"]["character_length"]["p95"]
        short_words = audit["text_length_summary"]["short_text_min_words"]
    context = {**inputs.feature_metadata.model_dump(mode="json"),
               "feature_configuration": inputs.feature_config.model_dump(mode="json"),
               "class_order": classes, "label_mapping": {label: i for i, label in enumerate(classes)},
               "evaluation_protocol": "evaluate_predictions; explicit class order; zero_division=0; validation selection, test descriptive only",
               "class_counts_full": counts, "top_k": top_k,
               "length_rule": {"measure": "character count of frozen subject + message text",
                               "short": "<= full-dataset nearest-rank p5 (ties included)",
                               "long": ">= full-dataset nearest-rank p95 (ties included)",
                               "short_cutoff": short_cutoff, "long_cutoff": long_cutoff,
                               "limited_context_rule": f"audit word count < {short_words}"},
               "experiments": {name: manifests[name]["experiment_id"] for name in MODELS},
               "engineering_environment": {name: {key: manifests[name][key] for key in ("hardware", "versions")} for name in MODELS},
               "source_references": {name: str(path) for name, path in directories.items()}}
    comparison, per_class, confusion, minority, length_groups, disagreements, reviews = {}, {}, {}, {}, {}, {}, {}
    pending_jsonl = {}
    for split in ("validation", "test"):
        records = getattr(inputs.splits, split)
        texts = {record.record_id: build_classification_text(record) for record in records}
        predictions, results, matrices, errors = {}, {}, {}, {}
        comparison[split] = {}
        for name, directory in directories.items():
            rows = [json.loads(line) for line in (directory / f"{split}_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            rows = _validate_predictions(rows, records, models[name], inputs.matrices[split], classes, name)
            predictions[name] = rows
            metrics, matrix = evaluate_predictions([row["actual_label"] for row in rows],
                                                   [row["predicted_label"] for row in rows], classes)
            if metrics != saved_metrics[name][split] or metrics != manifests[name]["evaluation_metrics"][split]:
                raise ValueError(f"{name}: incompatible evaluation protocol or metrics")
            if matrix != saved_confusion[name][split]:
                raise ValueError(f"{name}: confusion artifact differs")
            timing = manifests[name]["inference_latency"][split]
            if timing["record_count"] != len(records) or timing["clock"] != "perf_counter":
                raise ValueError(f"{name}: timing population/protocol differs")
            comparison[split][name] = {"model_name": manifests[name]["model_name"], **metrics,
                                      "training_time_seconds": manifests[name]["training_time"],
                                      "cpu_inference_seconds_per_record": timing["seconds_per_record"],
                                      "cpu_inference_metadata": timing,
                                      "model_size_bytes": manifests[name]["model_size_bytes"]}
            results[name], matrices[name] = metrics, matrix
            errors[name] = extract_errors(rows, records, name)
            pending_jsonl[f"{split}_{name}_errors.jsonl"] = errors[name]
        per_class[split] = compare_per_class(results, classes)
        confusion[split] = {name: {"confusion_matrix": matrices[name], "off_diagonal_pairs": confusion_pairs(matrices[name]),
                                   "per_class_errors": {label: {"false_positive_count": len(class_errors(errors[name], label, "false_positive")),
                                                                "false_negative_count": len(class_errors(errors[name], label, "false_negative"))}
                                                        for label in classes}} for name in MODELS}
        minority[split] = minority_analysis(counts, per_class[split], errors)
        length_groups[split] = length_analysis(records, predictions, short_cutoff, long_cutoff)
        differing = extract_disagreements(predictions[MODELS[0]], predictions[MODELS[1]])
        differing = [{**row, "text": texts[row["record_id"]]} for row in differing]
        disagreements[split] = {key: sum(row["category"] == key for row in differing) for key in
                                ("lr_correct_svm_wrong", "svm_correct_lr_wrong", "both_wrong_different")}
        pending_jsonl[f"{split}_model_disagreements.jsonl"] = differing
        ambiguous, label_issues = review_candidates(predictions[MODELS[0]], predictions[MODELS[1]], texts, short_words)
        reviews[split] = {"possible_ambiguity": len(ambiguous), "possible_label_issue": len(label_issues)}
        pending_jsonl[f"{split}_possible_ambiguous_examples.jsonl"] = ambiguous
        pending_jsonl[f"{split}_possible_label_issues.jsonl"] = label_issues
    features = {name: top_features(model, inputs.vectorizer, top_k) for name, model in models.items()}
    importance = {"top_k": top_k, "semantics": "Strongest signed model weights/associations, not causal explanations or probabilities",
                  "models": features, "overlap": feature_overlap(features, classes)}
    # Explicit evidence gate, not a claimed optimal weighting policy: imbalanced
    # classes plus a below-macro recall in a below-largest class on validation.
    largest = max(counts.values())
    weak = [label for label in classes if counts[label] < largest and any(
        comparison["validation"][name]["per_class"][label]["recall"] < comparison["validation"][name]["macro_recall"]
        for name in MODELS)]
    assessment = {"class_weighting_investigation_warranted": bool(weak), "evidence_classes": weak,
                  "policy": "Exploratory evidence gate: unequal frozen counts and below-macro validation recall in a less frequent class; not an optimization rule",
                  "reasoning": ("Less frequent classes with below-macro recall warrant a controlled Phase 2.9 investigation; inspect FP/FN trade-offs and label-review candidates first."
                                if weak else "No below-macro recall in less frequent validation classes; present evidence does not warrant weighting on this gate."),
                  "class_counts_full": counts,
                  "validation_evidence": {label: {name: {**comparison["validation"][name]["per_class"][label],
                                                         **confusion["validation"][name]["per_class_errors"][label]} for name in MODELS} for label in classes}}
    lr, svm = (comparison["validation"][name] for name in MODELS)
    selection_keys = ("macro_f1", "macro_recall", "macro_precision", "weighted_f1")
    lr_rank, svm_rank = (tuple(result[key] for key in selection_keys) for result in (lr, svm))
    winner = MODELS[1] if svm_rank > lr_rank else MODELS[0] if lr_rank > svm_rank else "tie"
    conclusion = {"preferred_baseline": winner, "selection_split": "validation",
                  "selection_metric_priority": list(selection_keys),
                  "svm_supporting_metric_improvements": [key for key in selection_keys if svm[key] > lr[key]],
                  "svm_class_f1_improvements": [row["class"] for row in per_class["validation"] if row["svm_minus_lr"]["f1"] > 0],
                  "reasoning": "Validation Macro F1 is the primary ranking criterion; ties use macro recall, macro precision, then weighted F1. Supporting metrics, per-class gains/regressions, minority behavior and engineering trade-offs qualify the recommendation below. Test is descriptive and does not tune the choice.",
                  "tradeoffs": {key: svm[key] - lr[key] for key in (*METRICS, "training_time_seconds", "cpu_inference_seconds_per_record", "model_size_bytes")},
                  "class_f1_regressions": [row["class"] for row in per_class["validation"] if row["svm_minus_lr"]["f1"] < 0],
                  "recall_tradeoffs": [row["class"] for row in per_class["validation"] if row["svm_minus_lr"]["recall"] < 0],
                  "largest_f1_gain_class": max(per_class["validation"], key=lambda row: row["svm_minus_lr"]["f1"])["class"],
                  "short_validation_error_rate_delta": length_groups["validation"]["short"][MODELS[1]]["error_rate"] - length_groups["validation"]["short"][MODELS[0]]["error_rate"],
                  "latency_limitation": "Recorded predict-only CPU times exclude TF-IDF; LR one cold pass versus SVM one warmup/five measured passes. Indicative only, not a controlled benchmark."}
    analysis = {"context": context, "overall": comparison, "per_class": per_class, "confusion": confusion,
                "minority": minority, "length_groups": length_groups, "disagreements": disagreements,
                "review_counts": reviews, "feature_importance": importance, "class_weighting_assessment": assessment,
                "conclusion": conclusion}
    analysis["analysis_id"] = "comparison_" + fingerprint_value(analysis)[:12]
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in pending_jsonl.items():
        _write_jsonl(output / filename, rows)
    for filename, value in (("classical_model_comparison.json", analysis), ("per_class_comparison.json", per_class),
                            ("confusion_analysis.json", confusion), ("feature_importance.json", importance)):
        _write_json(output / filename, value)
    _atomic_write_text(output / "classical_model_comparison.md", render_comparison_report(analysis))
    return analysis


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def render_comparison_report(analysis):
    """Generate all displayed numbers from the shared structured analysis."""
    a = analysis
    context = a["context"]
    parts = ["# Classical Model Comparison", "## Experiment Context",
             f"Dataset: `{context['dataset_version']}`. Split: `{context['split_version']}`. Features: `{context['feature_version']}`.",
             f"Compared models: TF-IDF + Logistic Regression and TF-IDF + Linear SVM. Input: {context['feature_configuration']['input_policy']}. Frozen labels: {', '.join(context['class_order'])}.",
             "TF-IDF configuration: `" + json.dumps(context["feature_configuration"], sort_keys=True) + "`.",
             context["evaluation_protocol"],
             "All records were joined by record_id and checked against frozen labels, model predictions and score outputs. Persisted metrics and confusion matrices were verified with the shared evaluator. No retraining or tuning occurred.",
             "## Overall Metrics"]
    for split in ("validation", "test"):
        parts += [f"### {split.title()}", _table(["Metric", "LR", "SVM"],
                    [[key, *(f"{a['overall'][split][name][key]:.6f}" for name in MODELS)] for key in METRICS])]
    parts += ["## Per-Class Performance"]
    for split in ("validation", "test"):
        parts += [f"### {split.title()}", _table(["Class", "Support", "LR P", "SVM P", "LR R", "SVM R", "LR F1", "SVM F1", "Δ F1 (SVM−LR)"],
                  [[row["class"], row[MODELS[0]]["support"],
                    *(f"{row[name][key]:.4f}" for key in ("precision", "recall", "f1") for name in MODELS),
                    f"{row['svm_minus_lr']['f1']:+.4f}"] for row in a["per_class"][split]]),
                  "Macro F1 difference is the arithmetic mean of the class F1 deltas; support does not weight this mean."]
    parts += ["## Confusion Matrix Analysis", "Rows are actual classes; columns are predicted classes, in the same frozen order."]
    for split in ("validation", "test"):
        for name in MODELS:
            matrix = a["confusion"][split][name]["confusion_matrix"]
            parts += [f"### {split.title()} — {name}", _table(["Actual / predicted", *context["class_order"]],
                     [[label, *row] for label, row in zip(context["class_order"], matrix["values"])])]
    parts += ["## Most Confused Class Pairs", "Rates are relative to actual-class support; diagonals are excluded."]
    for split in ("validation", "test"):
        parts += [f"### {split.title()}", _table(["Model", "Actual", "Predicted", "Count", "Rate"],
                  [[name, row["actual_class"], row["predicted_class"], row["error_count"], f"{row['error_rate']:.4f}"]
                   for name in MODELS for row in a["confusion"][split][name]["off_diagonal_pairs"][:5]])]
    for heading, kind in (("False Positive Analysis", "false_positive"), ("False Negative Analysis", "false_negative")):
        parts += [f"## {heading}", "Every misclassification is inspectable in the split/model errors JSONL with frozen input text, IDs and distinct probability/decision-score fields. Filter with class_errors(errors, label, kind).",
                  _table(["Split", "Class", "LR count", "SVM count"],
                         [[split, label, *(a["confusion"][split][name]["per_class_errors"][label][kind + "_count"] for name in MODELS)]
                          for split in ("validation", "test") for label in context["class_order"]])]
    parts += ["## Minority-Class Analysis", "Smallest classes are derived from frozen full-dataset class counts (all ties retained).",
              _table(["Class", "Full count"], sorted(context["class_counts_full"].items()))]
    for split in ("validation", "test"):
        parts += [f"### {split.title()}", _table(["Class", "Model", "Precision", "Recall", "F1", "FN", "FP"],
                  [[row["class"], name, *(f"{row[name][key]:.4f}" for key in ("precision", "recall", "f1")),
                    row[name]["false_negative_count"], row[name]["false_positive_count"]]
                   for row in a["minority"][split] for name in MODELS])]
        for row in a["minority"][split]:
            for name in MODELS:
                below = row[name]["recall"] < a["overall"][split][name]["macro_recall"]
                parts.append(f"{name}: {row['class']} recall is {'below' if below else 'at or above'} macro recall. Compare the full per-class table for weaknesses outside the smallest class.")
    parts += ["## Logistic Regression vs SVM Disagreements", _table(["Split", "Category", "Count"],
              [[split, key, value] for split, counts in a["disagreements"].items() for key, value in counts.items()]),
              "Details: validation_model_disagreements.jsonl and test_model_disagreements.jsonl; equal predictions are excluded.",
              "## Short / Long Ticket Analysis", "Rule: `" + json.dumps(context["length_rule"], sort_keys=True) + "`. Ties are included, so groups can exceed 5%; no input preprocessing changed.",
              _table(["Split", "Group", "N", "Model", "Errors", "Error rate", "Elevated vs all"],
                     [[split, group, data["record_count"], name, data[name]["error_count"],
                       f"{data[name]['error_rate']:.4f}" if data[name]["error_rate"] is not None else "N/A",
                       data[name].get("elevated_vs_all", "reference")]
                      for split, groups in a["length_groups"].items() for group, data in groups.items() for name in MODELS])]
    for heading, key, filename in (("Possible Ambiguous Tickets", "possible_ambiguity", "possible_ambiguous_examples"),
                                   ("Possible Label Issues", "possible_label_issue", "possible_label_issues")):
        parts += [f"## {heading}", _table(["Split", "Review candidates"], [[split, data[key]] for split, data in a["review_counts"].items()]),
                  f"Details: validation_{filename}.jsonl and test_{filename}.jsonl. Heuristic manual-review candidates only, not confirmed ambiguity or incorrect labels. Labels are never modified."]
    parts += ["Ambiguity indicators: both models wrong, differing predictions, or limited word context. Label-issue indicator: both models predict the same alternative to the frozen label. Review categories may overlap.",
              "## Feature Explainability", a["feature_importance"]["semantics"] + f". Top-k: {context['top_k']}. Full signed weights: feature_importance.json."]
    for heading, name in (("Logistic Regression", MODELS[0]), ("Linear SVM", MODELS[1])):
        parts += [f"### {heading}", _table(["Class", "Strongest positive associations", "Strongest negative associations"],
                  [[label, *( ", ".join(f"{row['feature']} ({row['weight']:.3f})" for row in a["feature_importance"]["models"][name][label][direction][:5])
                              for direction in ("positive", "negative"))] for label in context["class_order"]])]
    parts += ["### Shared / Different Important Features", "Overlap uses deterministic top-k sets per sign; differences do not establish causality.",
              _table(["Class", "Sign", "Shared", "LR only", "SVM only"],
                     [[label, direction, *( ", ".join(data[key]) for key in ("shared", "lr_only", "svm_only"))]
                      for label, signs in a["feature_importance"]["overlap"].items() for direction, data in signs.items()])]
    for heading, key, unit in (("Training Time", "training_time_seconds", "seconds"),
                               ("CPU Inference Latency", "cpu_inference_seconds_per_record", "seconds/record"),
                               ("Model Size", "model_size_bytes", "bytes")):
        parts += [f"## {heading}", _table(["Split", f"LR ({unit})", f"SVM ({unit})"],
                  [[split, *(f"{a['overall'][split][name][key]:.9g}" for name in MODELS)] for split in ("validation", "test")])]
        if heading == "CPU Inference Latency":
            parts += [a["conclusion"]["latency_limitation"], "Original timing metadata is retained in classical_model_comparison.json. No P50/P95 or throughput measurements were available; no re-benchmark was run."]
    parts += ["## Model Issues vs Data-Quality Issues", _table(["Category", "Interpretation"],
              [["Model error", "Prediction differs from frozen label; FP/FN artifacts"],
               ["Possible ambiguity", "Disagreement, both wrong, or limited context; heuristic review"],
               ["Possible label issue", "Shared alternative prediction; heuristic review, models can share bias"],
               ["Input-data limitation", "Short inputs/limited context can omit useful evidence; length analysis is descriptive"]]),
              "## Class Weighting Assessment", f"class_weighting_investigation_warranted = {str(a['class_weighting_assessment']['class_weighting_investigation_warranted']).lower()}",
              a["class_weighting_assessment"]["policy"], a["class_weighting_assessment"]["reasoning"],
              "Evidence classes: " + (", ".join(a["class_weighting_assessment"]["evidence_classes"]) or "none") + ". Full precision/recall/F1, FP/FN and frozen counts are retained in the structured assessment. No class weighting was applied.",
              "## Classical Baseline Conclusion", f"Preferred baseline: **{a['conclusion']['preferred_baseline']}**.", a["conclusion"]["reasoning"],
              "Supporting validation metrics higher for SVM: " + (", ".join(a["conclusion"]["svm_supporting_metric_improvements"]) or "none") + ". Classes with higher SVM F1: " + (", ".join(a["conclusion"]["svm_class_f1_improvements"]) or "none") + ".",
              "Validation differences (SVM minus LR): `" + json.dumps(a["conclusion"]["tradeoffs"], sort_keys=True) + "`.",
              "Classes with lower SVM validation F1: " + (", ".join(a["conclusion"]["class_f1_regressions"]) or "none") + ". Minority recall and error patterns above qualify this ranking; model size excludes the shared TF-IDF vectorizer. Engineering measurements are descriptive, with the timing caveat above.",
              "Largest validation F1 gain: " + a["conclusion"]["largest_f1_gain_class"] + ". Classes with lower SVM recall: " + (", ".join(a["conclusion"]["recall_tradeoffs"]) or "none") + ". SVM minus LR shortest-validation error rate: " + f"{a['conclusion']['short_validation_error_rate_delta']:+.4f}" + ". These trade-offs must remain visible despite the aggregate improvement.",
              "Next planned phase: Phase 2.7 — Transformer Dataset and Tokenization. No Transformer work is included here."]
    return "\n\n".join(parts) + "\n"

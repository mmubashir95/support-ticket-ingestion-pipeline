"""Deterministic classification metrics with explicit frozen label ordering."""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_predictions(actual, predicted, class_order):
    if not actual or len(actual) != len(predicted):
        raise ValueError("evaluation requires aligned nonempty predictions")
    if not class_order or len(set(class_order)) != len(class_order):
        raise ValueError("class order must contain unique labels")
    if not set(actual).union(predicted).issubset(class_order):
        raise ValueError("evaluation contains unknown labels")
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=class_order, zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        actual, predicted, labels=class_order, average="weighted", zero_division=0,
    )[2]
    metrics = {
        "class_order": list(class_order), "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "weighted_f1": float(weighted),
        "zero_division": 0,
        "per_class": {label: {"precision": float(p), "recall": float(r),
                              "f1": float(f), "support": int(s)}
                      for label, p, r, f, s in zip(class_order, precision, recall, f1, support)},
    }
    matrix = {"row_labels": list(class_order), "column_labels": list(class_order),
              "rows": "actual_label", "columns": "predicted_label",
              "values": confusion_matrix(actual, predicted, labels=class_order).tolist()}
    return metrics, matrix

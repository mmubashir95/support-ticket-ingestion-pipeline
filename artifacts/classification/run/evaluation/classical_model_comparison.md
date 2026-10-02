# Classical Model Comparison

## Experiment Context

Dataset: `ds_6fa2f19cf683`. Split: `split_d1f67d41500f`. Features: `tfidf_3275ba614aa6`.

Compared models: TF-IDF + Logistic Regression and TF-IDF + Linear SVM. Input: subject + message. Frozen labels: Change, Incident, Problem, Request.

TF-IDF configuration: `{"analyzer": "word", "feature_type": "tfidf", "input_fields": ["subject", "message"], "input_policy": "subject + message", "lowercase": true, "max_df": null, "max_features": null, "min_df": 2, "ngram_range": [1, 2], "separator": "\n\n", "sublinear_tf": false}`.

evaluate_predictions; explicit class order; zero_division=0; validation selection, test descriptive only

All records were joined by record_id and checked against frozen labels, model predictions and score outputs. Persisted metrics and confusion matrices were verified with the shared evaluator. No retraining or tuning occurred.

## Overall Metrics

### Validation

| Metric | LR | SVM |
| --- | --- | --- |
| macro_precision | 0.873004 | 0.888666 |
| macro_recall | 0.826383 | 0.867811 |
| macro_f1 | 0.837739 | 0.875505 |
| weighted_f1 | 0.835213 | 0.870374 |
| accuracy | 0.846315 | 0.874067 |

### Test

| Metric | LR | SVM |
| --- | --- | --- |
| macro_precision | 0.872427 | 0.885122 |
| macro_recall | 0.820843 | 0.864760 |
| macro_f1 | 0.831969 | 0.872058 |
| weighted_f1 | 0.829579 | 0.865559 |
| accuracy | 0.842854 | 0.869667 |

## Near-Duplicate Sensitivity

Each held-out ticket is matched to its most similar train ticket (max cosine of each held-out row to any train row in the frozen TF-IDF space). Buckets: near_copy >= 0.8, intermediate [0.6, 0.8), novel < 0.6; thresholds were fixed before reporting and are not tuned. The split, features and models are unchanged.

Headline metrics include held-out tickets with templated near copies in train; novel-ticket metrics are the conservative estimate for unseen ticket wording.

| Split | Bucket | N | NN label agreement | LR macro F1 | SVM macro F1 | LR accuracy | SVM accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | all | 4288 | 0.8979 | 0.8377 | 0.8755 | 0.8463 | 0.8741 |
| validation | near_copy | 404 | 1.0000 | 0.9382 | 0.9848 | 0.9431 | 0.9851 |
| validation | intermediate | 1074 | 0.9981 | 0.8929 | 0.9449 | 0.8985 | 0.9432 |
| validation | novel | 2810 | 0.8448 | 0.8020 | 0.8338 | 0.8125 | 0.8317 |
| test | all | 4289 | 0.8839 | 0.8320 | 0.8721 | 0.8429 | 0.8697 |
| test | near_copy | 403 | 1.0000 | 0.9240 | 0.9809 | 0.9454 | 0.9851 |
| test | intermediate | 1032 | 0.9961 | 0.8770 | 0.9378 | 0.8857 | 0.9370 |
| test | novel | 2854 | 0.8269 | 0.8061 | 0.8381 | 0.8129 | 0.8290 |

NN label agreement is the share of tickets whose nearest train ticket has the same label. Per-record evidence (ids, cosine, labels; no text): validation_nearest_train.jsonl and test_nearest_train.jsonl. A class absent from a bucket contributes F1 0 to that bucket's macro F1; per-class support is in the JSON.

## Per-Class Performance

### Validation

| Class | Support | LR P | SVM P | LR R | SVM R | LR F1 | SVM F1 | Δ F1 (SVM−LR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Change | 438 | 0.9879 | 0.9836 | 0.9315 | 0.9612 | 0.9589 | 0.9723 | +0.0134 |
| Incident | 1720 | 0.7594 | 0.8162 | 0.9233 | 0.9035 | 0.8334 | 0.8576 | +0.0242 |
| Problem | 902 | 0.7639 | 0.7670 | 0.4557 | 0.6131 | 0.5708 | 0.6815 | +0.1106 |
| Request | 1228 | 0.9807 | 0.9879 | 0.9951 | 0.9935 | 0.9879 | 0.9907 | +0.0028 |

Macro F1 difference is the arithmetic mean of the class F1 deltas; support does not weight this mean.

### Test

| Class | Support | LR P | SVM P | LR R | SVM R | LR F1 | SVM F1 | Δ F1 (SVM−LR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Change | 439 | 0.9951 | 0.9907 | 0.9339 | 0.9704 | 0.9636 | 0.9804 | +0.0169 |
| Incident | 1720 | 0.7513 | 0.8088 | 0.9308 | 0.8977 | 0.8315 | 0.8509 | +0.0195 |
| Problem | 902 | 0.7610 | 0.7507 | 0.4235 | 0.5942 | 0.5442 | 0.6634 | +0.1192 |
| Request | 1228 | 0.9823 | 0.9903 | 0.9951 | 0.9967 | 0.9887 | 0.9935 | +0.0048 |

Macro F1 difference is the arithmetic mean of the class F1 deltas; support does not weight this mean.

## Confusion Matrix Analysis

Rows are actual classes; columns are predicted classes, in the same frozen order.

### Validation — logistic_regression

| Actual / predicted | Change | Incident | Problem | Request |
| --- | --- | --- | --- | --- |
| Change | 408 | 13 | 0 | 17 |
| Incident | 0 | 1588 | 127 | 5 |
| Problem | 0 | 489 | 411 | 2 |
| Request | 5 | 1 | 0 | 1222 |

### Validation — linear_svm

| Actual / predicted | Change | Incident | Problem | Request |
| --- | --- | --- | --- | --- |
| Change | 421 | 8 | 0 | 9 |
| Incident | 0 | 1554 | 165 | 1 |
| Problem | 2 | 342 | 553 | 5 |
| Request | 5 | 0 | 3 | 1220 |

### Test — logistic_regression

| Actual / predicted | Change | Incident | Problem | Request |
| --- | --- | --- | --- | --- |
| Change | 410 | 13 | 1 | 15 |
| Incident | 0 | 1601 | 118 | 1 |
| Problem | 0 | 514 | 382 | 6 |
| Request | 2 | 3 | 1 | 1222 |

### Test — linear_svm

| Actual / predicted | Change | Incident | Problem | Request |
| --- | --- | --- | --- | --- |
| Change | 426 | 4 | 2 | 7 |
| Incident | 1 | 1544 | 175 | 0 |
| Problem | 1 | 360 | 536 | 5 |
| Request | 2 | 1 | 1 | 1224 |

## Most Confused Class Pairs

Rates are relative to actual-class support; diagonals are excluded.

### Validation

| Model | Actual | Predicted | Count | Rate |
| --- | --- | --- | --- | --- |
| logistic_regression | Problem | Incident | 489 | 0.5421 |
| logistic_regression | Incident | Problem | 127 | 0.0738 |
| logistic_regression | Change | Request | 17 | 0.0388 |
| logistic_regression | Change | Incident | 13 | 0.0297 |
| logistic_regression | Incident | Request | 5 | 0.0029 |
| linear_svm | Problem | Incident | 342 | 0.3792 |
| linear_svm | Incident | Problem | 165 | 0.0959 |
| linear_svm | Change | Request | 9 | 0.0205 |
| linear_svm | Change | Incident | 8 | 0.0183 |
| linear_svm | Problem | Request | 5 | 0.0055 |

### Test

| Model | Actual | Predicted | Count | Rate |
| --- | --- | --- | --- | --- |
| logistic_regression | Problem | Incident | 514 | 0.5698 |
| logistic_regression | Incident | Problem | 118 | 0.0686 |
| logistic_regression | Change | Request | 15 | 0.0342 |
| logistic_regression | Change | Incident | 13 | 0.0296 |
| logistic_regression | Problem | Request | 6 | 0.0067 |
| linear_svm | Problem | Incident | 360 | 0.3991 |
| linear_svm | Incident | Problem | 175 | 0.1017 |
| linear_svm | Change | Request | 7 | 0.0159 |
| linear_svm | Problem | Request | 5 | 0.0055 |
| linear_svm | Change | Incident | 4 | 0.0091 |

## False Positive Analysis

Every misclassification is inspectable in the split/model errors JSONL with frozen input text, IDs and distinct probability/decision-score fields. Filter with class_errors(errors, label, kind).

| Split | Class | LR count | SVM count |
| --- | --- | --- | --- |
| validation | Change | 5 | 7 |
| validation | Incident | 503 | 350 |
| validation | Problem | 127 | 168 |
| validation | Request | 24 | 15 |
| test | Change | 2 | 4 |
| test | Incident | 530 | 365 |
| test | Problem | 120 | 178 |
| test | Request | 22 | 12 |

## False Negative Analysis

Every misclassification is inspectable in the split/model errors JSONL with frozen input text, IDs and distinct probability/decision-score fields. Filter with class_errors(errors, label, kind).

| Split | Class | LR count | SVM count |
| --- | --- | --- | --- |
| validation | Change | 30 | 17 |
| validation | Incident | 132 | 166 |
| validation | Problem | 491 | 349 |
| validation | Request | 6 | 8 |
| test | Change | 29 | 13 |
| test | Incident | 119 | 176 |
| test | Problem | 520 | 366 |
| test | Request | 6 | 4 |

## Minority-Class Analysis

Smallest classes are derived from frozen full-dataset class counts (all ties retained).

| Class | Full count |
| --- | --- |
| Change | 2922 |
| Incident | 11466 |
| Problem | 6012 |
| Request | 8187 |

### Validation

| Class | Model | Precision | Recall | F1 | FN | FP |
| --- | --- | --- | --- | --- | --- | --- |
| Change | logistic_regression | 0.9879 | 0.9315 | 0.9589 | 30 | 5 |
| Change | linear_svm | 0.9836 | 0.9612 | 0.9723 | 17 | 7 |

logistic_regression: Change recall is at or above macro recall. Compare the full per-class table for weaknesses outside the smallest class.

linear_svm: Change recall is at or above macro recall. Compare the full per-class table for weaknesses outside the smallest class.

### Test

| Class | Model | Precision | Recall | F1 | FN | FP |
| --- | --- | --- | --- | --- | --- | --- |
| Change | logistic_regression | 0.9951 | 0.9339 | 0.9636 | 29 | 2 |
| Change | linear_svm | 0.9907 | 0.9704 | 0.9804 | 13 | 4 |

logistic_regression: Change recall is at or above macro recall. Compare the full per-class table for weaknesses outside the smallest class.

linear_svm: Change recall is at or above macro recall. Compare the full per-class table for weaknesses outside the smallest class.

## Logistic Regression vs SVM Disagreements

| Split | Category | Count |
| --- | --- | --- |
| validation | lr_correct_svm_wrong | 65 |
| validation | svm_correct_lr_wrong | 184 |
| validation | both_wrong_different | 7 |
| test | lr_correct_svm_wrong | 88 |
| test | svm_correct_lr_wrong | 203 |
| test | both_wrong_different | 5 |

Details: validation_model_disagreements.jsonl and test_model_disagreements.jsonl; equal predictions are excluded.

## Short / Long Ticket Analysis

Rule: `{"limited_context_rule": "audit word count < 3", "long": ">= full-dataset nearest-rank p95 (ties included)", "long_cutoff": 742, "measure": "character count of frozen subject + message text", "short": "<= full-dataset nearest-rank p5 (ties included)", "short_cutoff": 112}`. Ties are included, so groups can exceed 5%; no input preprocessing changed.

| Split | Group | N | Model | Errors | Error rate | Elevated vs all |
| --- | --- | --- | --- | --- | --- | --- |
| validation | all | 4288 | logistic_regression | 659 | 0.1537 | reference |
| validation | all | 4288 | linear_svm | 540 | 0.1259 | reference |
| validation | short | 221 | logistic_regression | 30 | 0.1357 | False |
| validation | short | 221 | linear_svm | 35 | 0.1584 | True |
| validation | long | 225 | logistic_regression | 33 | 0.1467 | False |
| validation | long | 225 | linear_svm | 24 | 0.1067 | False |
| test | all | 4289 | logistic_regression | 674 | 0.1571 | reference |
| test | all | 4289 | linear_svm | 559 | 0.1303 | reference |
| test | short | 200 | logistic_regression | 45 | 0.2250 | True |
| test | short | 200 | linear_svm | 38 | 0.1900 | True |
| test | long | 228 | logistic_regression | 40 | 0.1754 | True |
| test | long | 228 | linear_svm | 34 | 0.1491 | True |

## Possible Ambiguous Tickets

| Split | Review candidates |
| --- | --- |
| validation | 725 |
| test | 763 |

Details: validation_possible_ambiguous_examples.jsonl and test_possible_ambiguous_examples.jsonl. Heuristic manual-review candidates only, not confirmed ambiguity or incorrect labels. Labels are never modified.

## Possible Label Issues

| Split | Review candidates |
| --- | --- |
| validation | 468 |
| test | 466 |

Details: validation_possible_label_issues.jsonl and test_possible_label_issues.jsonl. Heuristic manual-review candidates only, not confirmed ambiguity or incorrect labels. Labels are never modified.

Ambiguity indicators: both models wrong, differing predictions, or limited word context. Label-issue indicator: both models predict the same alternative to the frozen label. Review categories may overlap.

## Feature Explainability

Strongest signed model weights/associations, not causal explanations or probabilities. Top-k: 20. Full signed weights: feature_importance.json.

### Logistic Regression

| Class | Strongest positive associations | Strongest negative associations |
| --- | --- | --- |
| Change | update (4.851), request (4.019), improve (3.941), enhance (3.736), aktualisierung (3.339) | problem (-2.736), have (-2.326), issue (-2.267), with (-1.911), due to (-1.894) |
| Incident | incident (3.143), issue (2.939), have (2.677), problem (2.632), the (2.505) | für (-3.022), for (-2.823), integration (-2.304), enhance (-2.168), request (-1.887) |
| Problem | problem (3.240), facing (2.400), despite (2.358), problems (2.233), due to (2.175) | for (-2.837), für (-2.089), tools (-2.070), zur (-1.650), data analytics (-1.618) |
| Request | on (4.519), details (4.041), you (3.939), für (3.558), for (3.475) | problem (-3.135), the (-2.819), update (-2.714), issue (-2.686), das (-2.404) |

### Linear SVM

| Class | Strongest positive associations | Strongest negative associations |
| --- | --- | --- |
| Change | update (2.945), aktualisierung (2.354), improve (2.317), enhance (2.234), verbessern (2.230) | detailed (-1.442), problem (-1.428), details (-1.350), on (-1.336), detaillierte (-1.318) |
| Incident | incident (2.699), initial (2.206), erste (1.938), error (1.879), several (1.788) | für (-1.671), data overload (-1.655), frequent (-1.635), updating (-1.628), updating the (-1.611) |
| Problem | trotz (2.078), frequent (1.888), facing (1.833), breaches (1.822), despite (1.797) | incident (-1.944), initial (-1.874), or hardware (-1.746), erste (-1.660), several (-1.615) |
| Request | details (2.492), detailed (2.305), on (2.082), detaillierte (2.038), könnten sie (1.866) | update (-2.004), aktualisierung (-1.656), problem (-1.655), issue (-1.589), issues (-1.537) |

### Shared / Different Important Features

Overlap uses deterministic top-k sets per sign; differences do not establish causality.

| Class | Sign | Shared | LR only | SVM only |
| --- | --- | --- | --- | --- |
| Change | positive | advanced, aktualisierung, an update, enhance, enhanced, enhancement, improve, request, revise, update, verbessern, verbesserung | and, current, for, our, to, tools, um, verbesserung der | beantragen, boost, enhancements, erweiterte, improved, optimierung, update der, upgrade |
| Change | negative | about, despite, details, due, due to, have, issue, könnten sie, on, problem, with, wurde, wurden | be, bei, breach, das problem, issues, problems, the | comprehensive, detailed, detaillierte, erfahren, geben, information on, you provide |
| Incident | positive | einen, error, incident, initial, issue, outage, several | campaign, das problem, has, hat, have, issues, problem, recent, software, system, the, wurde, wurden | access attempt, an incident, attempt, beeinträchtigt, crashes in, erste, kampagne, konflikte, multiple products, or hardware, restarting services, services the, vorfall |
| Incident | negative | breaches, enhance, für, request | data, details, for, ich, improve, integrating, integration, on, projektmanagement, sie, um, verbessern, verbesserung, would, you, zu | bigcommerce, client, cloud, crashes which, data overload, efficiency, frequent, kaspersky, launch, rechnung, requesting, security vulnerabilities, systems due, terraform, updating, updating the |
| Problem | positive | data access, despite, facing, frequent, trotz, updating | attempts, das, discrepancies, due, due to, have, issue, issues, outdated, problem, probleme, problems, the, with | breaches, client engagement, data overload, datenschutzbeschränkung, difficulties in, for security, ladezeiten, rechnung, security vulnerabilities, subscription, systems due, unstimmigkeiten, updating the, user load |
| Problem | negative | attempt, incident, initial, product, several | analytics, data analytics, enhance, for, für, ich, improve, informationen, integration, on, optimierung, produkte, tools, you, zur | access attempt, an incident, an unexpected, but have, crashes in, data synchronization, erste, multiple products, optimization system, or hardware, services the, take to, these, updates or, vorfall |
| Request | positive | bereitstellen, detailed, detaillierte, details, for, für, information on, informationen, integrating, könnten sie, on, provide, sie, you, zur | about, could you, integration, von, you provide | comprehensive, documentation, interessiert, können, könnten |
| Request | negative | aktualisierung, das, issue, issues, our, problem, probleme, problems, software, the, this, update, updates | das problem, der, die, due, due to, have, system | aufgrund, concern, enhanced, facing, implementing, improve, improved |

## Training Time

| Split | LR (seconds) | SVM (seconds) |
| --- | --- | --- |
| validation | 3.96374855 | 0.600932844 |
| test | 3.96374855 | 0.600932844 |

## Model Size

| Split | LR (bytes) | SVM (bytes) |
| --- | --- | --- |
| validation | 2680775 | 2680643 |
| test | 2680775 | 2680643 |

## CPU Inference Latency

| Split | Measure | LR | SVM |
| --- | --- | --- | --- |
| validation | Batch, text to label (µs/record) | 70.9 | 75.5 |
| validation | Batch throughput (records/s) | 14,110 | 13,241 |
| validation | Single ticket, text to label (ms) | 0.713 | 0.683 |
| validation | Classifier predict only (µs/record) | 0.462 | 0.436 |
| test | Batch, text to label (µs/record) | 72.3 | 77.4 |
| test | Batch throughput (records/s) | 13,829 | 12,917 |
| test | Single ticket, text to label (ms) | 0.672 | 0.828 |
| test | Classifier predict only (µs/record) | 0.431 | 0.679 |

Both models use the shared text_to_label_v1 protocol on the same machine: raw text -> TF-IDF transform -> predict, medians after warm-up. The shared TF-IDF transform dominates; predict-only time is shown separately. Single-machine medians, not the Phase 2.11 P50/P95 benchmark; small differences are within run-to-run noise.

## Model Issues vs Data-Quality Issues

| Category | Interpretation |
| --- | --- |
| Model error | Prediction differs from frozen label; FP/FN artifacts |
| Possible ambiguity | Disagreement, both wrong, or limited context; heuristic review |
| Possible label issue | Shared alternative prediction; heuristic review, models can share bias |
| Input-data limitation | Short inputs/limited context can omit useful evidence; length analysis is descriptive |

## Class Weighting Assessment

class_weighting_investigation_warranted = true

Exploratory evidence gate: unequal frozen counts and below-macro validation recall in a less frequent class; not an optimization rule

Less frequent classes with below-macro recall warrant a controlled Phase 2.9 investigation; inspect FP/FN trade-offs and label-review candidates first.

Evidence classes: Problem. Full precision/recall/F1, FP/FN and frozen counts are retained in the structured assessment. No class weighting was applied.

## Classical Baseline Conclusion

Preferred baseline: **linear_svm**.

Validation Macro F1 is the primary ranking criterion; ties use macro recall, macro precision, then weighted F1. Supporting metrics, per-class gains/regressions, minority behavior and engineering trade-offs qualify the recommendation below. Test is descriptive and does not tune the choice.

Supporting validation metrics higher for SVM: macro_f1, macro_recall, macro_precision, weighted_f1. Classes with higher SVM F1: Change, Incident, Problem, Request.

Validation differences (SVM minus LR): `{"accuracy": 0.027751865671641784, "cpu_inference_seconds_per_record": 4.647181553855945e-06, "cpu_single_record_latency_seconds": -3.0629500543000177e-05, "macro_f1": 0.03776548913255329, "macro_precision": 0.015662071371924324, "macro_recall": 0.04142804921424181, "model_size_bytes": -132, "training_time_seconds": -3.362815705993853, "weighted_f1": 0.03516086621965997}`.

Classes with lower SVM validation F1: none. Minority recall and error patterns above qualify this ranking; model size excludes the shared TF-IDF vectorizer. Engineering measurements are descriptive, with the timing caveat above.

Largest validation F1 gain: Problem. Classes with lower SVM recall: Incident, Request. SVM minus LR shortest-validation error rate: +0.0226. These trade-offs must remain visible despite the aggregate improvement.

Novel-ticket validation macro F1 (cosine < 0.6, n=2810): logistic_regression 0.8020, linear_svm 0.8338. Preferred-baseline ranking holds on novel tickets: true. Headline metrics include held-out tickets with templated near copies in train; novel-ticket metrics are the conservative estimate for unseen ticket wording.

Next planned phase: Phase 2.7 — Transformer Dataset and Tokenization. No Transformer work is included here.

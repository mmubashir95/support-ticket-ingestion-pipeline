# Phase 2.12 Final Three-Model Comparison

## Final recommendation

**Selected model: TF-IDF + Linear SVM.** TF-IDF + Linear SVM has the highest final-test Macro F1. Its per-class results, calibration evidence, and measured operational profile are reported together; missing benchmark cells are not estimated.

## Fair-comparison contract

All quality rows use the same frozen `test` split (4289 records), label order, selected Phase 2.9 model artifacts, and unchanged argmax predictions. The test set is now consumed and Phase 2 quality selection is frozen.

## Final comparison table

| Metric | TF-IDF + Logistic Regression | TF-IDF + Linear SVM | Transformer classifier |
|---|---:|---:|---:|
| Macro F1 | 0.8515 | 0.8724 | 0.8396 |
| Macro Precision | 0.8490 | 0.8772 | 0.8481 |
| Macro Recall | 0.8544 | 0.8689 | 0.8351 |
| Weighted F1 | 0.8418 | 0.8645 | 0.8327 |
| Training time (s) | 1.379 | 0.289 | 1207.854 |
| CPU P50 latency (ms, batch 1) | N/A | N/A | N/A |
| CPU P95 latency (ms, batch 1) | N/A | N/A | N/A |
| Accelerator P50 (ms, batch 1) | N/A | N/A | 3.613 |
| Model size (bytes) | 2680791 | 2680659 | 268783731 |
| Explainability | High | High | Medium/Low |
| Probability output | Native; temperature-scaled | Calibrated from margins | Native softmax; temperature-scaled |
| Calibrated ECE (validation evidence) | 0.0132 | 0.0222 | 0.0259 |
| Operational complexity | Low | Low | Higher |
| Cost per 1K | N/A | N/A | N/A |

## Per-class quality

| Class | TF-IDF + Logistic Regression | TF-IDF + Linear SVM | Transformer classifier |
|---|---:|---:|---:|
| Change F1 | 0.9604 | 0.9783 | 0.9646 |
| Incident F1 | 0.8075 | 0.8429 | 0.8137 |
| Problem F1 | 0.6495 | 0.6749 | 0.5877 |
| Request F1 | 0.9886 | 0.9935 | 0.9923 |

## Statistical and operational meaning

- **TF-IDF + Logistic Regression vs TF-IDF + Linear SVM:** paired Macro-F1 difference 95% CI `-0.0282` to `-0.0134`; exact McNemar p=`3.494e-11`.
- **TF-IDF + Logistic Regression vs Transformer classifier:** paired Macro-F1 difference 95% CI `0.0011` to `0.0232`; exact McNemar p=`0.5313`.
- **TF-IDF + Linear SVM vs Transformer classifier:** paired Macro-F1 difference 95% CI `0.0220` to `0.0443`; exact McNemar p=`1.627e-07`.

## Benchmark completeness

The Phase 2.11 archive is partial. Missing measurements are shown as N/A, never inferred:
- logistic_regression CPU P50/P95
- linear_svm CPU P50/P95
- transformer CPU P50/P95

## Decision questions

The recommendation considers Macro F1 and per-class behavior first, then latency, footprint, explainability, probability handling, deployment requirements, and cost availability. A Transformer is not preferred merely because it is newer.

## Threshold and calibration status

Calibration is included in confidence reporting, while argmax classes remain unchanged. Operational threshold status: `pending_business_acceptance_criterion`.

## Limitations

- The test split is used here for the final comparison and must not be used for further tuning.
- Calibration ECE is validation threshold-selection evidence, not a test-set estimate.
- Exact McNemar tests compare accuracy/correctness; paired bootstrap intervals compare Macro F1.
- No operational rejection threshold is selected because no business acceptance criterion was supplied.
- Infrastructure cost is unavailable unless an hourly rate was supplied to Phase 2.11.

## Phase 2 freeze

The selected artifacts, frozen split identities, metrics, and recommendation in this report form the Phase 2 handoff. Any later model change requires a new versioned evaluation rather than tuning against this test report.

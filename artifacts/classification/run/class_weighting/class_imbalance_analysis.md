# Phase 2.9 Class Imbalance Analysis

Dataset `ds_6fa2f19cf683`, split `split_d1f67d41500f`. Weights and frequencies use the training split only.

| Class | Training count | Percentage |
| --- | ---: | ---: |
| Change | 2045 | 10.22% |
| Incident | 8026 | 40.11% |
| Problem | 4208 | 21.03% |
| Request | 5731 | 28.64% |

Majority: **Incident**. Minority: **Change**. Ratio: **3.925:1**.

## Existing unweighted validation evidence

| Model | Status | Macro F1 |
| --- | --- | ---: |
| logistic_regression | available | 0.837739 |
| linear_svm | available | 0.875505 |
| distilbert | available | 0.841500 |

### logistic_regression

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Change | 0.987893 | 0.931507 | 0.958872 | 438 |
| Incident | 0.759445 | 0.923256 | 0.833377 | 1720 |
| Problem | 0.763941 | 0.455654 | 0.570833 | 902 |
| Request | 0.980738 | 0.995114 | 0.987874 | 1228 |

### linear_svm

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Change | 0.983645 | 0.961187 | 0.972286 | 438 |
| Incident | 0.816176 | 0.903488 | 0.857616 | 1720 |
| Problem | 0.766990 | 0.613082 | 0.681454 | 902 |
| Request | 0.987854 | 0.993485 | 0.990662 | 1228 |

### distilbert

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Change | 0.965986 | 0.972603 | 0.969283 | 438 |
| Incident | 0.781504 | 0.840116 | 0.809751 | 1720 |
| Problem | 0.644330 | 0.554324 | 0.595948 | 902 |
| Request | 0.993453 | 0.988599 | 0.991020 | 1228 |

## Decision

The frozen training distribution is unequal and available validation results show class-specific trade-offs. A controlled experiment is warranted, but weighting is not presumed to improve validation Macro F1. Frequency alone does not establish causation.

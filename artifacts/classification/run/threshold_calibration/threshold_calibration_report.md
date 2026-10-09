# Phase 2.10 Threshold Tuning and Calibration

Dataset `ds_6fa2f19cf683`, split `split_d1f67d41500f`.

## Prediction audit

- linear_svm: available
- logistic_regression: available
- transformer: available

## Calibration policy

- linear_svm: `sigmoid_calibration`
- logistic_regression: `sigmoid_calibration`
- transformer: `temperature_scaling`

## Threshold policy

- linear_svm: selected; threshold=0.8
- logistic_regression: selected; threshold=0.8
- transformer: selected; threshold=0.85

## Methodological limitation

The frozen validation split previously influenced model/checkpoint selection. The internal calibration-fitting and threshold-selection subsets are disjoint, but they are not fully independent test data. No final test predictions are used.

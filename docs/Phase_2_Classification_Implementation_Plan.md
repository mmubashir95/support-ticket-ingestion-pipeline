# Phase 2 — Sparse Text Representations and Modern Classification

## Goal

Phase 2 moves the project from a completed ingestion pipeline into supervised text classification.

The objective is to build and compare:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Transformer-based classifier

The comparison must be fair and reproducible. All three models should use the same dataset definition, label policy, train/validation/test split, and evaluation protocol.

---

# Phase 2 Final Outcome

At the end of Phase 2, the project should be able to:

- Load the accepted, cleaned Phase 1 dataset.
- Define a clear classification target.
- Create reproducible train, validation, and test splits.
- Build TF-IDF features.
- Train a Logistic Regression classifier.
- Train a Linear SVM classifier.
- Train or fine-tune a Transformer classifier.
- Handle class imbalance appropriately.
- Evaluate all models using the same metrics.
- Tune thresholds where appropriate.
- Evaluate probability calibration where applicable.
- Benchmark batch inference.
- Measure CPU and GPU latency where relevant.
- Measure model size.
- Compare explainability and operational cost.
- Produce a final model comparison report.
- Select the most suitable model for production use.

---

# Phase 2 Principles

## Principle 1 — Reuse Phase 1

Phase 2 must consume the accepted output produced by the Phase 1 ingestion pipeline.

Do not reimplement:

- cleaning
- normalization
- validation
- PII masking
- deduplication
- language detection
- leakage checks

Typical flow:

```text
Raw source data
    ↓
Phase 1 ingestion pipeline
    ↓
accepted.jsonl
    ↓
Phase 2 classification dataset
```

---

## Principle 2 — Use one frozen evaluation protocol

All models must use:

```text
Same input tickets
Same labels
Same train split
Same validation split
Same test split
Same evaluation metrics
```

This is required for a meaningful comparison.

---

## Principle 3 — Build baselines before Transformers

The order should be:

```text
Dataset understanding
↓
TF-IDF
↓
Logistic Regression
↓
Linear SVM
↓
Classical-model evaluation
↓
Transformer
↓
Advanced evaluation
↓
Final comparison
```

Do not start with the Transformer.

---

# Phase 2 Structure

Phase 2 is divided into 12 implementation parts.

---

# Phase 2.1 — Classification Dataset Audit

## Objective

Understand exactly what will be predicted before building a model.

## Questions to answer

- What is the target label?
- Is the task binary, multi-class, or multi-label?
- How many examples are available?
- How many classes exist?
- How many examples exist per class?
- Are labels missing?
- Are labels invalid or inconsistent?
- Are there very rare classes?
- Is the dataset imbalanced?
- What is the text length distribution?
- Which Phase 1 fields will be used as model input?
- Are there duplicate or conflicting labels?
- Are there possible leakage fields?

## Recommended initial scope

Prefer a single-label multi-class problem for the first classifier if the dataset supports it.

Example:

```text
Ticket                              Label
------------------------------------------------
"I was charged twice"               Billing
"I cannot log in"                   Account Issue
"My package has not arrived"        Delivery
"I want to cancel my subscription"  Cancellation
```

## Implementation work

Create a dataset audit module/report that produces:

- total records
- usable records
- missing label count
- class count
- per-class frequency
- class percentage
- rare-class summary
- ticket text length summary
- duplicate-label conflicts
- input field summary

## Deliverable

```text
phase2_dataset_audit.md
```

or equivalent generated report.

## Exit criteria

- Classification target is explicitly defined.
- Task type is known.
- Class distribution is documented.
- Input text fields are frozen.
- Leakage-prone fields are excluded.
- Dataset limitations are documented.

---

# Phase 2.2 — Train / Validation / Test Split

## Objective

Create one reproducible dataset split that all models will use.

## Recommended default

For a normal IID classification dataset:

```text
Train       70%
Validation  15%
Test        15%
```

or:

```text
Train       80%
Validation  10%
Test        10%
```

The exact percentage is less important than reproducibility.

## Important rule

Do not fit TF-IDF before splitting.

Wrong:

```text
Entire dataset
↓
Fit TF-IDF
↓
Split dataset
```

Correct:

```text
Dataset
↓
Split
↓
Fit TF-IDF on train only
↓
Transform validation
↓
Transform test
```

## Stratification

For multi-class classification, use stratification where practical so class proportions remain similar across splits.

## Reproducibility

Store:

- random seed
- split policy
- record identifiers
- split counts
- class counts for each split

## Deliverable

A frozen split artifact or deterministic split procedure.

Example:

```text
artifacts/phase2/splits/
    train.jsonl
    validation.jsonl
    test.jsonl
    split_manifest.json
```

## Exit criteria

- No train/test leakage.
- Split can be reproduced.
- Class distributions are documented.
- Same split can be reused by all models.

---

# Phase 2.3 — TF-IDF Feature Pipeline

## Objective

Convert ticket text into sparse numerical features.

## Concepts to understand

Before implementation, understand:

- Bag of Words
- term frequency
- inverse document frequency
- TF-IDF
- sparse matrices
- word n-grams
- character n-grams
- vocabulary
- out-of-vocabulary behavior

## Baseline configuration

Start simple.

Example baseline:

```text
word n-grams: (1, 2)
lowercase: true
min_df: sensible small value
max_df: optional
sublinear_tf: optional
max_features: optional
```

Do not heavily tune in the first implementation.

## Input text policy

Decide whether classification input is:

```text
subject
```

or:

```text
subject + body
```

or another explicitly documented combination.

Keep this identical across classical models.

## Persist

Store:

- fitted vectorizer
- vocabulary size
- feature configuration
- training metadata

## Deliverable

Reusable TF-IDF feature pipeline.

## Exit criteria

- TF-IDF is fitted only on training data.
- Validation/test are transformed using the fitted vectorizer.
- Sparse matrices are used.
- Feature configuration is versioned.
- Vocabulary statistics are documented.

---

# Phase 2.4 — Logistic Regression Baseline

## Objective

Train the first strong probabilistic classical classifier.

## Pipeline

```text
Ticket text
↓
TF-IDF
↓
Logistic Regression
↓
Class scores / probabilities
↓
Predicted class
```

## Why Logistic Regression first?

It provides:

- strong text-classification baseline
- probabilities
- interpretable coefficients
- fast CPU inference
- support for class weighting
- compatibility with threshold tuning
- compatibility with calibration analysis

## Initial implementation

Use a simple baseline first.

Do not immediately run a large hyperparameter search.

Record:

- solver
- regularization
- C
- max iterations
- class weighting
- random seed

## Evaluation

At minimum:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- per-class precision
- per-class recall
- per-class F1
- confusion matrix
- training time
- inference latency
- model size

## Deliverable

```text
TF-IDF + Logistic Regression baseline
```

## Exit criteria

- Model trains reproducibly.
- Predictions are generated for validation/test.
- Metrics are saved.
- Probabilities are available.
- Model artifact can be reloaded for inference.

---

# Phase 2.5 — Linear SVM Baseline

## Objective

Train a second strong sparse-text classifier.

## Pipeline

```text
Ticket text
↓
TF-IDF
↓
Linear SVM
↓
Decision score
↓
Predicted class
```

## Important conceptual difference

Logistic Regression naturally produces probabilities.

Linear SVM typically produces decision scores, not calibrated probabilities.

Do not interpret raw SVM margins as probabilities.

## Fair comparison

Initially use the same:

- dataset
- split
- text input
- TF-IDF representation

This isolates the classifier difference.

## Evaluation

Use the same metrics as Logistic Regression.

Also record:

- training time
- CPU latency
- serialized size
- decision-score behavior

## Deliverable

```text
TF-IDF + Linear SVM baseline
```

## Exit criteria

- Model trains reproducibly.
- Same evaluation protocol is used.
- Validation/test predictions exist.
- Metrics and model artifacts are persisted.

---

# Phase 2.6 — Classical Model Evaluation and Error Analysis

## Objective

Compare Logistic Regression and Linear SVM before introducing a Transformer.

## Metrics

Compare:

- Macro F1
- Macro Precision
- Macro Recall
- Weighted F1
- Per-class F1
- Confusion matrix
- Training time
- CPU inference latency
- Model size

## Why Macro F1?

If the data is imbalanced:

```text
General Question  7000
Billing           1500
Cancellation       800
Fraud              200
```

accuracy may hide poor minority-class performance.

Macro F1 gives each class equal importance in the final average.

## Error analysis

Inspect:

- most confused classes
- common false positives
- common false negatives
- minority-class failures
- ambiguous tickets
- mislabeled examples
- unusually short or long tickets
- cases where LR and SVM disagree

## Explainability

For classical models, inspect:

- strongest positive features for each class
- strongest negative features
- important words/ngrams

## Deliverable

```text
classical_model_comparison.md
```

## Exit criteria

- Best classical baseline identified.
- Main error categories documented.
- Data-quality issues separated from model issues.
- Need for class weighting is understood.

---

# Phase 2.7 — Transformer Dataset and Tokenization

## Objective

Prepare the exact same task for Transformer fine-tuning.

## Concepts to understand

Before implementation:

- pretrained encoder
- Transformer tokenizer
- subword tokenization
- attention mask
- truncation
- padding
- maximum sequence length
- classification head
- fine-tuning

## Dataset policy

Reuse:

- same records
- same labels
- same train split
- same validation split
- same test split

Do not create a different dataset for the Transformer.

## Analyze ticket lengths

Before choosing `max_length`, inspect tokenized sequence length distribution.

For example:

```text
50th percentile
90th percentile
95th percentile
99th percentile
maximum
```

Choose truncation based on evidence rather than guessing.

## Model selection

Choose a practical pretrained encoder based on:

- language
- dataset size
- CPU/GPU constraints
- latency target
- model size
- expected deployment environment

Start with a relatively small encoder rather than a very large model.

## Deliverable

Transformer-ready tokenized datasets and documented tokenizer policy.

## Exit criteria

- Model choice documented.
- Tokenizer choice documented.
- Maximum sequence length justified.
- Train/validation/test alignment verified.

---

# Phase 2.8 — Transformer Fine-Tuning

## Objective

Fine-tune the pretrained Transformer for the same classification target.

## Pipeline

```text
Ticket text
↓
Tokenizer
↓
Token IDs + attention mask
↓
Pretrained encoder
↓
Classification head
↓
Fine-tuning
↓
Class logits
↓
Probability / class prediction
```

## Record training configuration

At minimum:

- pretrained model
- tokenizer
- epochs
- learning rate
- batch size
- weight decay
- random seed
- max sequence length
- early stopping policy
- class weighting if used
- hardware

## Save

- model
- tokenizer
- label mapping
- training configuration
- best checkpoint information
- training metrics

## Evaluate

Use exactly the same final test set as classical models.

## Exit criteria

- Transformer fine-tuning is reproducible.
- Best validation checkpoint is selected correctly.
- Test set is not used for model selection.
- Final predictions/metrics are saved.

---

# Phase 2.9 — Class Imbalance and Class Weighting

## Objective

Determine whether imbalance handling improves minority-class performance.

## First establish an unweighted baseline

Do not apply class weighting automatically.

Compare:

```text
Unweighted model
vs
Weighted model
```

## Classical models

Test class weighting where justified:

```text
class_weight="balanced"
```

or an explicitly calculated weighting strategy.

## Transformer

If necessary, use weighted cross-entropy or an equivalent justified strategy.

## Evaluate impact on

- macro F1
- minority-class recall
- minority-class precision
- overall error trade-offs

## Important lesson

Class weighting changes the cost of mistakes during training.

It does not directly define the final decision threshold.

## Deliverable

Class-weighting comparison report.

## Exit criteria

- Need for weighting is supported by evidence.
- Weighted vs unweighted results are compared.
- Final weighting policy is frozen.

---

# Phase 2.10 — Threshold Tuning and Calibration

## Objective

Separate three related concepts:

```text
Model score/probability
↓
Calibration
↓
Decision threshold
↓
Final prediction
```

## Threshold tuning

Use validation data to determine whether a non-default threshold improves the desired business metric.

This is especially relevant for:

- binary classification
- one-vs-rest decisions
- multi-label classification
- abstention/confidence policies

For standard single-label multi-class softmax classification, class selection is usually argmax, so threshold tuning may instead be used for:

- confidence rejection
- escalation
- "unknown" handling

## Calibration

Evaluate whether probability values match observed frequencies.

Example:

```text
Predictions around 0.8 confidence
→ approximately 80% should be correct
```

Possible metrics:

- calibration curve
- Brier score
- Expected Calibration Error if implemented carefully

## Logistic Regression

Often reasonably calibrated, but verify.

## SVM

Raw margins are not probabilities.

Only calibrate if probabilities are actually required.

## Transformer

Evaluate calibration because neural classifiers can be overconfident.

## Important split rule

Use validation data for:

- threshold selection
- calibration fitting

Use test data only for final evaluation.

## Deliverable

Threshold/calibration evaluation report.

## Exit criteria

- Threshold policy documented.
- Calibration need documented.
- Test set remains untouched until final evaluation.

---

# Phase 2.11 — Batch Inference and Performance Benchmarking

## Objective

Measure engineering performance, not just model quality.

## Benchmark

For each relevant model:

- single-example CPU latency
- batch CPU latency
- GPU latency for Transformer if GPU is available
- throughput
- batch size impact
- serialized model size
- memory usage where practical

## Latency methodology

Use:

- warm-up runs
- repeated measurements
- median / P50
- P95 where practical

Do not report only one timing measurement.

## Cost

For local/open models, cost can be represented by:

- CPU/GPU requirement
- memory requirement
- runtime
- infrastructure complexity

For hosted inference, also estimate:

- API cost
- cost per 1K tickets
- cost per 1M tickets

## Deliverable

```text
inference_benchmark.md
```

## Exit criteria

- Classical CPU latency recorded.
- Transformer CPU latency recorded where feasible.
- Transformer GPU latency recorded if GPU exists.
- Model sizes recorded.
- Batch behavior documented.

---

# Phase 2.12 — Final Three-Model Comparison

## Objective

Compare all three approaches and select the most appropriate model.

## Models

```text
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Transformer classifier
```

## Final comparison table

Example structure:

| Metric | TF-IDF + LR | TF-IDF + SVM | Transformer |
|---|---:|---:|---:|
| Macro F1 | | | |
| Macro Precision | | | |
| Macro Recall | | | |
| Weighted F1 | | | |
| Training Time | | | |
| CPU P50 Latency | | | |
| CPU P95 Latency | | | |
| GPU Latency | N/A | N/A | |
| Model Size | | | |
| Explainability | High | High | Medium/Low |
| Probability Output | Native | No/Optional | Native |
| Calibration Quality | | | |
| Operational Complexity | Low | Low | Higher |
| Cost | Low | Low | Higher |

## Final decision questions

Do not select the Transformer only because it is modern.

Ask:

- How much better is Macro F1?
- Which classes improve?
- Which classes become worse?
- Is the improvement statistically/operationally meaningful?
- How much slower is inference?
- Does deployment require a GPU?
- Is explainability important?
- Are probabilities needed?
- Is model size acceptable?
- What is the cost per prediction?
- How difficult is monitoring and deployment?

## Example engineering decision

If:

```text
Linear SVM Macro F1 = 0.89
Transformer Macro F1 = 0.90
```

but:

```text
SVM CPU latency = 2 ms
Transformer CPU latency = 80 ms
```

the SVM may be the better production choice.

The objective is not:

> Choose the most sophisticated model.

The objective is:

> Choose the simplest model that satisfies the business and quality requirements.

## Deliverable

```text
phase2_final_model_comparison.md
```

## Exit criteria

- Three models compared fairly.
- Test data used only for final comparison.
- Quality and engineering metrics included.
- Final model recommendation justified.
- Limitations documented.
- Phase 2 can be frozen.

---

# Recommended Repository Structure

Do not restructure working Phase 1 code unnecessarily.

One possible organization:

```text
src/
├── ticket_pipeline/
│   ├── loaders.py
│   ├── normalization.py
│   ├── pii.py
│   ├── pipeline.py
│   ├── outputs.py
│   └── ...
│
└── ticket_classification/
    ├── dataset.py
    ├── splitting.py
    ├── features.py
    ├── logistic_regression.py
    ├── svm.py
    ├── transformer.py
    ├── evaluation.py
    ├── calibration.py
    ├── benchmarking.py
    └── artifacts.py

tests/
├── ...
└── classification/
    ├── test_dataset.py
    ├── test_splitting.py
    ├── test_features.py
    ├── test_classical_models.py
    ├── test_evaluation.py
    ├── test_calibration.py
    └── test_benchmarking.py

artifacts/
├── phase1/
└── phase2/
    ├── dataset/
    ├── splits/
    ├── tfidf/
    ├── logistic_regression/
    ├── svm/
    ├── transformer/
    ├── evaluation/
    └── reports/

docs/
├── phase1/
└── phase2/
```

This is only a suggested structure. Prefer the repository's existing conventions over unnecessary reorganization.

---

# Phase 2 Evaluation Contract

Freeze a common evaluation contract before model comparison.

## Quality metrics

Required:

```text
Macro Precision
Macro Recall
Macro F1
Weighted F1
Per-class Precision
Per-class Recall
Per-class F1
Confusion Matrix
```

Optional:

```text
Accuracy
Micro F1
Top-k accuracy
```

Use optional metrics only when appropriate.

---

# Performance Metrics

Record:

```text
Training time
CPU P50 inference latency
CPU P95 inference latency
GPU inference latency where applicable
Throughput
Serialized model size
Memory use where practical
```

---

# Explainability Comparison

## Logistic Regression

High explainability.

Inspect feature coefficients by class.

## Linear SVM

High explainability for sparse text.

Inspect feature weights/margins.

## Transformer

Lower direct interpretability.

Use careful error analysis rather than treating attention as a definitive explanation.

---

# Cost Comparison

## Classical models

Usually:

```text
CPU inference
small model
low memory
low serving complexity
```

## Transformer

Potentially:

```text
higher CPU latency
GPU may be useful
larger model
higher memory
more deployment complexity
```

Cost should be evaluated as an engineering constraint, not only monetary API cost.

---

# Testing Strategy

Each sub-phase should add tests.

## Dataset tests

- label validation
- missing labels
- class mapping
- deterministic loading

## Split tests

- no overlap
- deterministic split
- class preservation where expected

## TF-IDF tests

- fit only on train
- deterministic vocabulary/config
- correct transform shapes

## Model tests

- model trains
- predictions have correct shape
- unknown labels rejected
- serialization/reload works

## Evaluation tests

- known confusion-matrix examples
- metric correctness
- missing-class handling

## Benchmark tests

Do not assert strict timing values in normal unit tests.

Test benchmark machinery separately from actual performance expectations.

---

# Artifact and Experiment Tracking

Every experiment should record enough information to reproduce it.

Recommended metadata:

```text
experiment_id
dataset_version
split_version
model_type
model_name
feature_config
hyperparameters
random_seed
training_environment
training_time
evaluation_metrics
model_size
timestamp
```

Avoid adding a heavy experiment-tracking platform unless needed.

A deterministic JSON/Markdown experiment record is sufficient initially.

---

# Learning Workflow

Do not implement all of Phase 2 in one Codex prompt.

For every sub-phase:

```text
1. Study the concept
2. Explain the concept in your own words
3. Prepare an implementation plan
4. Review repository impact
5. Ask Codex to implement only that sub-phase
6. Review the implementation yourself
7. Run tests
8. Inspect outputs
9. Write what you learned
10. Move to the next sub-phase
```

This is important because the goal is to learn AI engineering, not only to produce code using AI.

---

# Recommended Execution Order

```text
Phase 2.1
Dataset Audit
    ↓
Phase 2.2
Train / Validation / Test Split
    ↓
Phase 2.3
TF-IDF Feature Pipeline
    ↓
Phase 2.4
Logistic Regression
    ↓
Phase 2.5
Linear SVM
    ↓
Phase 2.6
Classical Evaluation + Error Analysis
    ↓
Phase 2.7
Transformer Dataset + Tokenization
    ↓
Phase 2.8
Transformer Fine-Tuning
    ↓
Phase 2.9
Class Weighting
    ↓
Phase 2.10
Threshold Tuning + Calibration
    ↓
Phase 2.11
Batch Inference + Benchmarking
    ↓
Phase 2.12
Final Comparison + Model Selection
```

---

# Suggested Milestones

## Milestone A — Dataset Ready

Complete:

- Phase 2.1
- Phase 2.2

Result:

```text
Frozen classification dataset and evaluation split
```

---

## Milestone B — Classical Baselines Ready

Complete:

- Phase 2.3
- Phase 2.4
- Phase 2.5
- Phase 2.6

Result:

```text
TF-IDF + Logistic Regression
TF-IDF + Linear SVM
Classical model comparison
```

---

## Milestone C — Transformer Ready

Complete:

- Phase 2.7
- Phase 2.8

Result:

```text
Fine-tuned Transformer classifier
```

---

## Milestone D — Production Evaluation Ready

Complete:

- Phase 2.9
- Phase 2.10
- Phase 2.11

Result:

```text
Imbalance policy
Threshold policy
Calibration analysis
Inference benchmarks
```

---

## Milestone E — Phase 2 Complete

Complete:

- Phase 2.12

Result:

```text
Three-model production comparison
Final model recommendation
```

---

# What Not to Do Yet

Avoid:

- RAG
- vector databases
- agents
- fine-tuning LLMs
- production API serving
- Docker deployment
- complex hyperparameter optimization
- full MLOps platform
- feature-store architecture

Those belong to later phases.

Keep Phase 2 focused on:

```text
Text representation
Classification
Evaluation
Inference comparison
Model selection
```

---

# Definition of Done — Phase 2

Phase 2 is complete when all of the following are true:

```text
[ ] Classification target is frozen
[ ] Dataset audit completed
[ ] Train/validation/test split frozen
[ ] Leakage checks completed for classification
[ ] TF-IDF feature pipeline implemented
[ ] Logistic Regression trained
[ ] Linear SVM trained
[ ] Classical models evaluated
[ ] Transformer tokenizer policy defined
[ ] Transformer classifier fine-tuned
[ ] Class imbalance policy evaluated
[ ] Threshold policy evaluated where relevant
[ ] Probability calibration evaluated where relevant
[ ] Batch inference benchmarked
[ ] CPU latency measured
[ ] GPU latency measured where hardware is available
[ ] Model sizes measured
[ ] Explainability compared
[ ] Cost/operational complexity compared
[ ] All three models evaluated on the same test set
[ ] Final model comparison report produced
[ ] Final production recommendation documented
[ ] Tests passing
[ ] Documentation current
```

---

# Final Phase 2 Deliverables

At the end of Phase 2, the repository should contain evidence for:

```text
1. Dataset audit
2. Frozen data split
3. TF-IDF feature pipeline
4. Logistic Regression model
5. Linear SVM model
6. Transformer classifier
7. Classical model comparison
8. Class-imbalance experiment
9. Threshold/calibration analysis
10. Batch-inference benchmark
11. Final three-model comparison
12. Final production model recommendation
```

---

# Immediate Next Step

Start only with:

```text
Phase 2.1 — Classification Dataset Audit
```

Do not train Logistic Regression, SVM, or a Transformer yet.

The first implementation should establish:

```text
What are we predicting?
What data do we have?
How balanced is it?
What input fields are allowed?
What is the classification task type?
```

Once Phase 2.1 is complete, use its findings to design Phase 2.2.

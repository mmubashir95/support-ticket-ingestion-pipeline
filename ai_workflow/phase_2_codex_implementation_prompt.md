# Phase 2 — Codex Implementation Prompt

## Project

**Repository:** `support-ticket-ingestion-pipeline`

## Objective

Implement the production-style support-ticket classification layer on top of the completed ingestion pipeline.

The goal is to take the trusted, cleaned output produced by the ingestion system and build a fair, reproducible comparison of:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Transformer-based text classifier

The implementation must support reliable dataset preparation, reproducible splits, model training, evaluation, benchmarking, artifact persistence, and final model comparison.

The result should be useful as a real small-team machine-learning codebase while remaining easy for a human developer to understand, debug, maintain, and extend.

---

# Core Engineering Rules

These rules are mandatory.

## 1. Keep the implementation simple

Prefer straightforward Python and scikit-learn / Hugging Face patterns over framework-heavy abstractions.

Do not introduce:

- model registries,
- plugin architectures,
- dependency-injection containers,
- generic training frameworks,
- unnecessary factories,
- abstract base model hierarchies,
- workflow engines,
- orchestration platforms,

unless the existing repository already has a strong current need.

A developer should be able to trace:

```text
accepted dataset
→ classification dataset
→ split
→ features/tokenization
→ model training
→ predictions
→ evaluation
→ persisted artifacts
```

without navigating unnecessary abstraction layers.

## 2. Do not over-engineer

Build only what the current classification comparison requires.

Do not implement architecture for hypothetical future needs.

Avoid:

- distributed training,
- Kubernetes,
- feature stores,
- model registries,
- vector databases,
- online serving APIs,
- queues,
- microservices,
- full MLOps platforms,
- large hyperparameter search systems.

## 3. Reuse the completed ingestion pipeline

The existing ingestion pipeline is the trusted source of model-ready ticket data.

Do not reimplement:

- raw source loading policies,
- normalization,
- Unicode handling,
- HTML handling,
- PII masking,
- deduplication,
- language detection,
- dataset versioning,
- general leakage checks.

Classification must consume the accepted output produced by the ingestion layer.

## 4. Keep experiments reproducible

Every training run should make it possible to identify:

- dataset version,
- split version,
- model type,
- feature/tokenizer configuration,
- hyperparameters,
- random seed,
- training time,
- evaluation metrics,
- artifact locations.

Do not add a heavy experiment platform unless already present and justified.

Simple deterministic JSON/Markdown metadata is enough.

## 5. Separate model development from final evaluation

Use:

```text
training data
→ fit model

validation data
→ model selection / threshold / calibration decisions

test data
→ final comparison only
```

Do not repeatedly inspect or optimize against the test set.

## 6. Make debugging easy

Prefer:

- explicit control flow,
- typed configuration,
- clear model artifacts,
- deterministic splits,
- clear evaluation functions,
- concise logging,
- readable reports.

Avoid broad exception swallowing.

Never use:

```python
except Exception:
    pass
```

---

# Intended Scope

Implement the following capabilities:

1. Classification dataset contract and audit.
2. Classification-specific field/input policy.
3. Classification target validation.
4. Reproducible train/validation/test splitting.
5. Split manifest and leakage protection.
6. TF-IDF feature extraction.
7. TF-IDF + Logistic Regression classifier.
8. TF-IDF + Linear SVM classifier.
9. Shared evaluation utilities.
10. Per-class metrics.
11. Confusion-matrix reporting.
12. Classical-model explainability using feature weights.
13. Transformer dataset preparation.
14. Transformer tokenizer handling.
15. Transformer classifier fine-tuning.
16. Class-imbalance evaluation.
17. Class-weighting experiments where justified.
18. Threshold-policy evaluation where relevant.
19. Probability-calibration evaluation where relevant.
20. Batch-inference benchmarking.
21. CPU latency benchmarking.
22. GPU latency benchmarking when suitable hardware is available.
23. Serialized model-size measurement.
24. Training-time measurement.
25. Operational-cost / complexity comparison.
26. Final three-model comparison report.
27. Final production model recommendation.
28. Automated tests for high-value deterministic behavior.
29. Documentation for running and reproducing experiments.

---

# Explicitly Out of Scope

Do not implement:

- RAG,
- vector search,
- embeddings for retrieval,
- LLM response generation,
- ticket summarization,
- entity extraction,
- agent systems,
- tool use,
- model-serving API,
- FastAPI,
- Docker deployment,
- Kubernetes,
- streaming inference,
- distributed training,
- foundation-model pretraining,
- LoRA/QLoRA,
- reinforcement learning,
- large hyperparameter sweeps,
- automated AutoML.

The work ends when the three classifiers have been evaluated fairly and a model recommendation is documented.

---

# Data Source

Use the accepted output from the completed ingestion pipeline.

Expected source conceptually:

```text
accepted.jsonl
```

or the actual repository-equivalent accepted-record artifact.

Do not train from raw CSV directly if that bypasses the ingestion contract.

The classification layer should retain traceability to the dataset manifest/fingerprint where available.

---

# Recommended Repository Organization

Prefer the repository's current conventions.

A reasonable structure may resemble:

```text
src/
├── ticket_pipeline/
│   └── ... existing ingestion modules ...
│
└── ticket_classification/
    ├── __init__.py
    ├── dataset.py
    ├── audit.py
    ├── splitting.py
    ├── features.py
    ├── classical.py
    ├── transformer.py
    ├── evaluation.py
    ├── calibration.py
    ├── benchmarking.py
    ├── artifacts.py
    └── workflow.py
```

Tests may live under:

```text
tests/classification/
```

Artifacts may live under:

```text
artifacts/classification/
```

Do not move stable ingestion code merely to match this diagram.

Do not create modules that have no meaningful responsibility.

---

# Implementation Order

Implement incrementally in the order below.

---

## 1. Classification Dataset Contract and Audit

Before training anything, inspect the accepted-record schema and actual dataset.

Determine:

- target field,
- classification task type,
- usable input text fields,
- excluded/leakage-risk fields,
- total labeled records,
- missing labels,
- blank labels,
- unique classes,
- class counts,
- class percentages,
- rare classes,
- text-length distribution,
- conflicting labels for identical model inputs,
- suspicious label inconsistencies.

Possible task types:

```text
binary
single_label_multiclass
multilabel
```

Do not guess the task type.

If no valid supervised target exists, report the blocker clearly and do not fabricate one.

### Input text policy

Define one deterministic representation.

For example:

```text
subject + body
```

if supported by the actual schema.

Only use fields available at prediction time.

Do not include post-resolution or target-derived information.

### Output

Produce structured and human-readable audit artifacts.

Example:

```text
classification_dataset_audit.json
classification_dataset_audit.md
```

Use the repository's actual artifact conventions.

---

## 2. Freeze the Classification Split

After confirming the dataset is suitable for classification, create one reproducible train/validation/test split.

Use one common split for all three model families.

A reasonable default for sufficiently sized IID data is:

```text
train: 70%
validation: 15%
test: 15%
```

or:

```text
train: 80%
validation: 10%
test: 10%
```

Choose based on dataset size.

### Requirements

- deterministic random seed,
- stratification where appropriate,
- no record overlap,
- no duplicate-group leakage if duplicates/groups exist,
- test set isolated from model selection,
- split counts recorded,
- class distributions recorded.

### Important

If semantically or exactly related records are grouped, prevent group leakage across splits where practical.

### Persist

Example:

```text
train.jsonl
validation.jsonl
test.jsonl
split_manifest.json
```

The manifest should include:

- dataset fingerprint/version,
- split configuration,
- random seed,
- record counts,
- class counts,
- split fingerprints if practical.

---

## 3. TF-IDF Feature Pipeline

Implement reusable sparse feature extraction.

Understand and document:

- vocabulary,
- TF-IDF,
- sparse matrices,
- word n-grams,
- character n-grams if used.

Start with a simple word-level baseline.

A reasonable baseline may use:

```text
ngram_range = (1, 2)
lowercase = true
min_df = small sensible value
```

but choose based on the actual dataset and document the choice.

### Critical rule

Fit the vectorizer on the training set only.

Correct:

```text
train
→ fit TF-IDF

validation
→ transform using fitted TF-IDF

test
→ transform using fitted TF-IDF
```

Never fit TF-IDF on the full dataset before splitting.

### Persist

Persist:

- fitted vectorizer,
- feature configuration,
- vocabulary size,
- artifact metadata.

---

## 4. Logistic Regression Classifier

Train the first strong classical baseline.

Pipeline:

```text
classification text
→ TF-IDF
→ Logistic Regression
→ probabilities
→ predicted class
```

Use a simple initial configuration.

Record:

- solver,
- regularization,
- C,
- max iterations,
- class weighting,
- random seed.

Do not start with a large hyperparameter sweep.

### Required outputs

- trained model,
- predictions,
- probability output where supported,
- metrics,
- training duration,
- serialized model size,
- inference benchmarks,
- model metadata.

---

## 5. Linear SVM Classifier

Train a second sparse-text baseline using the same dataset and initial TF-IDF representation where reasonable.

Pipeline:

```text
classification text
→ TF-IDF
→ Linear SVM
→ decision score
→ predicted class
```

### Important

Raw SVM decision scores are not probabilities.

Do not label them as probabilities.

Only add probability calibration later if there is a real downstream need.

### Required outputs

- trained model,
- predictions,
- decision scores,
- metrics,
- training duration,
- model size,
- inference benchmarks,
- metadata.

---

## 6. Shared Evaluation Contract

Implement one evaluation layer used consistently across model families.

At minimum report:

```text
macro precision
macro recall
macro F1
weighted F1
per-class precision
per-class recall
per-class F1
support
confusion matrix
```

Accuracy may also be reported but must not be the sole quality metric.

### Macro F1

Macro F1 is a primary comparison metric because support-ticket classes may be imbalanced.

### Output

Persist machine-readable metrics and a human-readable report.

Example:

```text
metrics.json
evaluation.md
confusion_matrix.csv
```

Avoid making image/chart generation a hard requirement unless the repository already supports it cleanly.

---

## 7. Classical Model Error Analysis

Compare Logistic Regression and Linear SVM before implementing the Transformer.

Inspect:

- strongest classes,
- weakest classes,
- most confused class pairs,
- false positives,
- false negatives,
- minority-class failures,
- very short ticket failures,
- long/complex ticket failures,
- disagreements between LR and SVM,
- suspicious labels.

Do not automatically change labels during error analysis.

Report likely:

```text
model issue
data issue
label issue
ambiguous example
```

where evidence supports the distinction.

---

## 8. Classical Model Explainability

For TF-IDF + Logistic Regression and TF-IDF + Linear SVM, inspect learned feature weights.

Produce per-class top influential features where mathematically appropriate.

Example:

```text
Billing:
charged
invoice
payment failed
refund

Account:
password
login
reset
locked
```

Do not present feature weights as causal explanations.

The goal is model inspection/debugging.

Persist a concise explainability artifact.

---

## 9. Transformer Dataset Preparation

Reuse exactly the same:

- record IDs,
- labels,
- train split,
- validation split,
- test split.

Do not create a new convenient split for the Transformer.

### Model selection

Choose a practical pretrained encoder based on:

- dataset language,
- dataset size,
- ticket length,
- available hardware,
- model size,
- inference latency,
- deployment constraints.

Prefer a small/medium encoder suitable for text classification rather than a very large model.

Document the reason for the selected model.

### Token length audit

Using the actual tokenizer, calculate:

```text
median token length
P90
P95
P99
maximum
```

Use this evidence to select `max_length`.

Do not blindly use `512`.

---

## 10. Transformer Fine-Tuning

Fine-tune the selected pretrained encoder for the same target.

Pipeline:

```text
classification text
→ tokenizer
→ input IDs + attention mask
→ pretrained encoder
→ classification head
→ logits
→ class probabilities/predictions
```

Record:

- model identifier,
- tokenizer identifier,
- label mapping,
- epochs,
- batch size,
- learning rate,
- weight decay,
- max sequence length,
- random seed,
- early stopping policy,
- class weighting if used,
- training hardware.

### Model selection

Use validation performance to choose the best checkpoint.

Do not choose the checkpoint based on test results.

### Persist

- model,
- tokenizer,
- label mapping,
- training config,
- training metrics,
- best checkpoint metadata.

---

## 11. Class Imbalance Evaluation

Do not apply class weighting automatically.

First establish unweighted baselines.

Then evaluate whether imbalance handling improves results.

For classical models, test a justified weighting strategy such as:

```python
class_weight="balanced"
```

where appropriate.

For the Transformer, use weighted loss only if justified by the dataset and baseline results.

Compare:

- macro F1,
- macro recall,
- minority-class recall,
- minority-class precision,
- weighted F1,
- trade-offs on majority classes.

Do not oversample or use synthetic text unless separately justified.

Persist a concise imbalance experiment report.

---

## 12. Threshold Policy

Apply threshold tuning only where it makes sense.

### Binary / multi-label

Thresholds may directly determine class activation.

### Single-label multi-class

Normal class selection is usually:

```text
argmax(probabilities)
```

Threshold analysis may instead support:

- abstention,
- human-review routing,
- low-confidence handling,
- unknown/reject policies.

Do not force threshold tuning onto a problem where no useful threshold policy exists.

Use validation data for threshold selection.

Do not optimize thresholds on test data.

---

## 13. Probability Calibration

Evaluate calibration only for models that expose meaningful probabilities or where probabilities are operationally needed.

### Logistic Regression

Evaluate its probability calibration.

### Linear SVM

Do not treat decision scores as probabilities.

Only apply calibration if probability output is actually required.

### Transformer

Evaluate calibration because neural classifiers may be overconfident.

Possible measures:

- Brier score,
- reliability/calibration table,
- calibration curve data,
- Expected Calibration Error if implemented carefully.

Use validation data for calibration fitting.

Use test data only for final assessment.

Do not add unnecessary calibration complexity if the final product only requires class labels.

---

## 14. Batch Inference

Implement a simple, reusable batch inference path for all three models.

Measure behavior across representative batch sizes.

Do not build an online serving system.

The goal is to understand operational performance.

---

## 15. Performance Benchmarking

Measure:

```text
training time
single-example CPU latency
batch CPU latency
P50 latency
P95 latency
throughput
serialized model size
```

For the Transformer, additionally measure GPU latency if suitable GPU hardware is available.

If no GPU is available, report:

```text
GPU benchmark not executed
```

Do not fabricate numbers.

### Benchmark methodology

Use:

- warm-up runs,
- multiple measured iterations,
- consistent hardware,
- consistent batch definitions.

Do not assert strict latency thresholds in unit tests.

---

## 16. Cost / Operational Complexity Comparison

Compare operational characteristics.

### Classical models

Evaluate:

- CPU suitability,
- small model footprint,
- serving simplicity,
- explainability,
- retraining simplicity.

### Transformer

Evaluate:

- CPU performance,
- GPU benefit,
- memory,
- artifact size,
- serving complexity,
- training cost.

If no hosted service is used, do not invent API pricing.

Cost may be represented using infrastructure/resource requirements.

---

## 17. Final Model Comparison

Produce one final comparison of:

```text
TF-IDF + Logistic Regression
TF-IDF + Linear SVM
Transformer classifier
```

Use the same final test set.

Required comparison dimensions:

| Dimension | LR | SVM | Transformer |
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
| Explainability | | | |
| Calibration | | | |
| Operational Complexity | | | |
| Cost | | | |

### Decision principle

Do not automatically recommend the Transformer.

Select the simplest model that satisfies quality and operational requirements.

For example:

```text
SVM Macro F1 = 0.89
Transformer Macro F1 = 0.90
```

If the Transformer is dramatically slower/larger, the SVM may be the better production choice.

The final recommendation must explicitly discuss the trade-off.

---

# Artifact Tracking

Each experiment should be traceable.

Record metadata such as:

```text
experiment_id
dataset_version
split_version
model_type
model_name
feature/tokenizer config
hyperparameters
random_seed
training_time
metrics
model_size
hardware
```

Use simple JSON/Markdown unless the repository already uses another lightweight convention.

---

# Logging

Use concise logging.

Good examples:

```text
Loaded 18,420 labeled records
Created train/validation/test split: 12,894 / 2,763 / 2,763
Fitted TF-IDF vocabulary: 41,281 features
Trained Logistic Regression in 4.8s
Transformer validation Macro F1: 0.8731
```

Do not log raw ticket text or sensitive values unnecessarily.

---

# Testing Requirements

Use pytest.

Do not write tests that require long Transformer training.

Separate deterministic unit/integration tests from expensive real-model smoke tests where appropriate.

## Dataset tests

Test:

- target validation,
- input construction,
- missing labels,
- class counting,
- rare-class reporting,
- label inconsistency reporting,
- duplicate/conflicting labels.

## Split tests

Test:

- deterministic splits,
- no record overlap,
- class distribution preservation where possible,
- no grouped leakage if grouping is implemented,
- stable manifest creation.

## TF-IDF tests

Test:

- vectorizer fits on training data only,
- transform output shape,
- sparse output,
- deterministic configuration,
- serialization/reload.

## Classical model tests

Test:

- training succeeds on a small fixture,
- prediction shape,
- probability/score shape,
- serialization/reload,
- label mapping.

## Evaluation tests

Use controlled known examples to verify:

- macro precision,
- macro recall,
- macro F1,
- per-class metrics,
- confusion matrix.

## Calibration tests

Test deterministic calculation using small controlled probability arrays.

Do not require real long-running model training.

## Benchmark utilities

Test benchmark code mechanics.

Do not assert machine-specific latency values.

## Transformer tests

Use lightweight smoke/unit tests.

Do not download a large model in the normal test suite unless the dependency is already intentionally available.

Real transformer integration may remain opt-in if necessary.

## Integration test

Create at least one controlled end-to-end classification workflow covering:

```text
accepted records
→ dataset preparation
→ split
→ TF-IDF
→ Logistic Regression
→ evaluation
→ persisted artifacts
```

A separate lightweight Transformer smoke path may be used if practical.

---

# Error Handling

Differentiate:

### System/configuration errors

Examples:

```text
missing accepted dataset
invalid target configuration
missing trained model artifact
unsupported classifier
unwritable artifact directory
```

Fail clearly.

### Data-quality issues

Examples:

```text
missing target
rare class
label conflict
class imbalance
```

Report through the dataset audit/readiness system.

Do not silently discard important data without recording why.

---

# Documentation Requirements

Update the README or classification documentation with:

1. classification objective,
2. accepted dataset dependency,
3. target/input contract,
4. classification task type,
5. how to run dataset audit,
6. how the split is created,
7. how TF-IDF is built,
8. how classical models are trained,
9. how Transformer training works,
10. evaluation metrics,
11. class imbalance policy,
12. threshold/calibration policy,
13. benchmarking methodology,
14. artifact structure,
15. how to reproduce experiments,
16. known limitations,
17. explicit out-of-scope functionality.

Keep documentation practical.

---

# Implementation Workflow

Before editing code:

1. inspect the current repository,
2. inspect the accepted-record schema,
3. inspect the actual classification target,
4. inspect available dependencies,
5. identify existing artifact/versioning conventions,
6. reuse completed ingestion functionality,
7. define the smallest clean classification architecture.

Implement incrementally.

After each meaningful unit:

```bash
pytest
```

At the end, run the complete test suite.

Do not weaken tests to make them pass.

---

# Completion Criteria

The classification implementation is complete when:

```text
[ ] Classification target is explicitly defined
[ ] Task type is documented
[ ] Model input fields are frozen
[ ] Classification-specific leakage risks are documented
[ ] Dataset audit is implemented
[ ] Dataset readiness is documented
[ ] Train/validation/test split is reproducible
[ ] Split leakage protections are verified
[ ] TF-IDF feature pipeline is implemented
[ ] Logistic Regression is trained/evaluated
[ ] Linear SVM is trained/evaluated
[ ] Classical-model error analysis exists
[ ] Classical-model feature inspection exists
[ ] Transformer tokenizer/model policy is documented
[ ] Transformer classifier is fine-tuned
[ ] Class-weighting need is evaluated
[ ] Threshold policy is evaluated where relevant
[ ] Calibration is evaluated where relevant
[ ] Batch inference works
[ ] CPU latency is measured
[ ] GPU latency is measured when hardware permits
[ ] Model sizes are measured
[ ] Training times are measured
[ ] All three models use the same test set
[ ] Final comparison report exists
[ ] Final model recommendation is justified
[ ] Automated tests pass
[ ] Existing ingestion tests remain passing
[ ] Documentation is current
```

---

# Final Codex Response

At completion, provide:

## 1. Summary

Briefly explain what was implemented.

## 2. Repository Inspection

State what existing ingestion/data/versioning functionality was reused.

## 3. Files Changed

For each file:

```text
File:
Purpose:
Key behavior:
```

## 4. Classification Data Contract

Report:

```text
Target:
Task type:
Input fields:
Excluded/leakage fields:
Dataset version:
Usable labeled records:
Class count:
```

## 5. Split

Report:

```text
Train:
Validation:
Test:
Seed:
Stratification/grouping policy:
```

## 6. Models Implemented

Report configuration for:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Transformer classifier

## 7. Evaluation Results

Provide the actual comparison table.

Do not invent missing hardware metrics.

## 8. Class Imbalance / Threshold / Calibration

Explain what was evaluated and what was selected.

## 9. Performance Benchmark

Report:

```text
training time
CPU P50
CPU P95
throughput
GPU latency if available
model size
```

## 10. Tests

Provide:

```text
test command
passed
failed
skipped
```

Separate optional expensive-model smoke tests when appropriate.

## 11. Persisted Artifacts

List generated artifacts and their purpose.

## 12. Limitations

Separate:

```text
BLOCKERS
WARNINGS
OPTIONAL FUTURE IMPROVEMENTS
```

## 13. Final Recommendation

Choose the recommended production classifier and explain why using:

```text
quality
latency
size
explainability
calibration/probability needs
operational complexity
cost
```

Do not recommend the Transformer merely because it is more sophisticated.

## 14. Completion Status

Return exactly one:

```text
CLASSIFICATION IMPLEMENTATION COMPLETE
```

or:

```text
CLASSIFICATION IMPLEMENTATION NOT COMPLETE
```

Do not claim completion if required tests fail or the final model comparison has not been performed.

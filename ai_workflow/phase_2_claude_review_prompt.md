# Phase 2 — Claude Code Review Prompt

## Role

Act as a **senior NLP / machine-learning engineer reviewing a production-style text classification implementation**.

Review the classification work in the repository:

```text
support-ticket-ingestion-pipeline
```

The ingestion layer is already complete.

Your task is to determine whether the classification implementation is:

- correct,
- reproducible,
- scientifically fair,
- simple,
- modular,
- human-manageable,
- easy to debug,
- appropriately tested,
- aligned with practical industry ML engineering,
- and appropriately scoped for a learning/portfolio project.

This is not a request to redesign the repository into a large MLOps platform.

---

# Critical Review Principle

The implementation should follow:

> **The simplest reproducible design that supports a fair production-oriented comparison of classical and Transformer text classifiers.**

Reject both extremes:

```text
throwaway notebook experiment
```

and:

```text
unnecessary MLOps / enterprise over-engineering
```

The desired result is:

```text
simple
+ reproducible
+ leakage-safe
+ testable
+ debuggable
+ measurable
+ reusable
```

---

# Intended Classification Scope

The implementation should cover:

1. Classification dataset contract.
2. Classification dataset audit.
3. Target validation.
4. Prediction-time input field policy.
5. Classification-specific leakage checks.
6. Reproducible train/validation/test split.
7. TF-IDF representation.
8. Logistic Regression classifier.
9. Linear SVM classifier.
10. Shared evaluation.
11. Macro precision/recall/F1.
12. Per-class metrics.
13. Confusion matrix.
14. Classical feature inspection.
15. Transformer tokenizer preparation.
16. Transformer classifier fine-tuning.
17. Class imbalance evaluation.
18. Class weighting where justified.
19. Threshold-policy evaluation where relevant.
20. Calibration evaluation where relevant.
21. Batch inference.
22. CPU latency measurement.
23. GPU latency measurement when hardware is available.
24. Training-time measurement.
25. Model-size measurement.
26. Explainability comparison.
27. Cost/operational-complexity comparison.
28. Final three-model comparison.
29. Final model recommendation.
30. Automated tests.
31. Reproducibility documentation.

---

# Explicitly Out of Scope

Flag unnecessary implementation of:

- RAG,
- vector databases,
- semantic search,
- LLM response generation,
- agent systems,
- fine-tuning generative LLMs,
- FastAPI serving,
- Docker/Kubernetes deployment,
- model registries,
- feature stores,
- workflow orchestration platforms,
- distributed training,
- large AutoML systems,
- excessive hyperparameter optimization.

These should not be required for approval of the current classification implementation.

---

# Models Under Review

The required comparison is:

```text
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Transformer-based classifier
```

All three must be compared as fairly as practical.

---

# Review Areas

Review each area separately.

---

## 1. Repository / Architecture

Evaluate whether the classification code is cleanly separated from the completed ingestion code.

Check:

- Is ingestion reused instead of duplicated?
- Are model-development responsibilities easy to locate?
- Is the classification package cohesive?
- Is code over-fragmented?
- Is there a god module?
- Are abstractions justified by current needs?
- Can a developer trace the complete training flow?

A healthy flow should be understandable as:

```text
accepted dataset
→ audit
→ frozen split
→ representation/tokenization
→ train
→ validate
→ final test
→ benchmark
→ compare
```

Do not require a specific file tree if the existing one is clear.

---

## 2. Data Contract

Verify that the classifier uses the trusted accepted output of the ingestion pipeline.

Check:

- target field is explicit,
- task type is explicit,
- model input text is deterministic,
- source record ID is retained,
- dataset version/fingerprint is retained where available,
- excluded fields are documented.

Flag training directly from raw source data if that bypasses the ingestion contract without justification.

---

## 3. Classification Dataset Audit

Check whether the implementation reports:

- total records,
- usable labeled records,
- missing targets,
- blank targets,
- unique classes,
- per-class counts,
- per-class percentages,
- rare classes,
- text length,
- suspicious label formatting,
- duplicate model inputs,
- conflicting labels,
- leakage risks,
- readiness status.

Verify the audit reports problems instead of silently "fixing" them.

Flag automatic merging/relabeling without explicit business rules.

---

## 4. Prediction-Time Feature Safety

This is critical.

Verify that model inputs only use fields available at prediction time.

Potential leakage examples may include:

```text
resolved_at
final_status
assigned_team
resolution
agent_response
post-resolution notes
```

The actual assessment must use real repository fields.

Ask:

> Would this field exist when the production model is expected to predict?

If no, it should not be used as a model feature unless explicitly justified.

Treat confirmed label leakage as a blocker.

---

## 5. Split Correctness

Review the train/validation/test split carefully.

Check:

- split happens before fitting TF-IDF,
- deterministic seed,
- stratification when appropriate,
- no record overlap,
- test set is isolated,
- validation is used for model selection,
- class counts are recorded,
- split metadata is reproducible.

### Duplicate/group leakage

If duplicate or near-duplicate groups exist, inspect whether related records can leak across train/test.

If this risk is material and ignored, classify severity accordingly.

---

## 6. Test-Set Discipline

Explicitly inspect for test-set leakage.

Flag:

- hyperparameter choices based repeatedly on test results,
- threshold selection on test data,
- calibration fitting on test data,
- checkpoint selection using test metrics,
- choosing TF-IDF configuration by test performance.

Expected:

```text
train → fitting
validation → decisions
test → final unbiased comparison
```

Test-set misuse is a major or blocker-level issue depending on impact.

---

## 7. TF-IDF Pipeline

Review:

- fitted on training data only,
- validation/test only transformed,
- sparse representation preserved,
- configuration documented,
- vocabulary statistics recorded,
- vectorizer serialized/reloadable.

Check whether preprocessing duplicates destructive cleaning already handled by ingestion.

Do not require aggressive stop-word removal/stemming/lemmatization unless justified.

---

## 8. Logistic Regression

Review:

- appropriate classifier setup,
- convergence,
- regularization,
- solver,
- deterministic configuration where applicable,
- probability handling,
- class weighting policy,
- serialization,
- evaluation.

Check whether `predict_proba` outputs are interpreted correctly.

---

## 9. Linear SVM

Review:

- correct linear SVM implementation,
- same baseline data and TF-IDF where appropriate,
- decision scores handled correctly,
- no claim that raw SVM scores are probabilities,
- class weighting policy,
- serialization,
- evaluation.

Flag any probability-based calibration/threshold logic that treats raw margins as probabilities without calibration.

---

## 10. Fair Classical Comparison

Check whether LR vs SVM comparison changes only justified variables.

Initially they should use:

```text
same records
same split
same input representation
same evaluation metrics
```

If radically different preprocessing is used, require clear justification.

---

## 11. Evaluation Metrics

At minimum expect:

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

Accuracy may be included but should not dominate the decision.

Verify:

- averaging mode is correct,
- zero-division behavior is handled,
- labels are consistently ordered,
- metrics use correct ground truth/predictions,
- no train metrics are presented as final quality.

---

## 12. Macro F1 Usage

Check whether Macro F1 is treated as a primary metric when class imbalance exists.

Ensure the implementation does not choose a model purely on accuracy if minority classes perform poorly.

Review per-class results before approving the final recommendation.

---

## 13. Error Analysis

Evaluate whether the project inspects actual failure patterns.

Useful categories include:

- most confused classes,
- false positives,
- false negatives,
- minority-class errors,
- ambiguous tickets,
- likely annotation issues,
- short-text failures,
- long-text failures,
- disagreement between models.

Do not require huge manual analysis.

A focused, evidence-based error report is sufficient.

---

## 14. Classical Explainability

Review top-weighted TF-IDF features for LR/SVM.

Check:

- mapping from coefficients to features is correct,
- per-class handling is correct,
- signs/weights are interpreted correctly,
- explanation is described as model evidence, not causation.

Flag misleading explainability claims.

---

## 15. Transformer Model Selection

Review why the specific pretrained encoder was chosen.

Consider:

- language,
- ticket length,
- dataset size,
- hardware,
- model size,
- CPU/GPU latency.

Flag use of an unnecessarily large model without justification.

Prefer practical encoder classifiers over oversized generative models.

---

## 16. Transformer Tokenization

Review:

- correct tokenizer for the selected model,
- token-length audit,
- evidence-based `max_length`,
- truncation policy,
- padding policy,
- label mapping.

Flag blind `max_length=512` if most data is far shorter and no rationale is given.

---

## 17. Transformer Fine-Tuning

Review:

- train/validation/test reuse,
- checkpoint selection on validation,
- learning rate,
- batch size,
- epochs,
- weight decay,
- random seed,
- early stopping,
- label mapping,
- loss function,
- checkpoint persistence.

Check whether test data influences training decisions.

---

## 18. Class Imbalance Handling

Verify that class weighting was evaluated rather than assumed.

Expected sequence:

```text
unweighted baseline
→ inspect imbalance/per-class metrics
→ weighted experiment if justified
→ compare trade-offs
```

Check whether weighting improves minority performance without unacceptable degradation.

Do not require weighting when data does not justify it.

Flag oversampling/synthetic augmentation added without need.

---

## 19. Threshold Policy

Check whether threshold tuning is appropriate for the actual task.

### Binary / multi-label

Threshold selection may be directly relevant.

### Single-label multi-class

Default class prediction is normally argmax.

A threshold may instead support:

- abstention,
- low-confidence escalation,
- human review.

Flag meaningless threshold tuning performed only because it was on a checklist.

Ensure validation, not test, was used to choose thresholds.

---

## 20. Calibration

Review whether calibration is applied meaningfully.

Check:

- probabilities vs raw scores are distinguished,
- Brier score/calibration calculations are correct,
- calibration fitting uses validation data,
- final calibration evaluation uses held-out test data,
- calibration is not added where no probability-dependent product decision exists.

For SVM, raw margins must not be presented as calibrated probabilities.

---

## 21. Batch Inference

Review whether batch inference is implemented simply and consistently.

Check:

- correct preprocessing/tokenization,
- correct label mapping,
- stable output structure,
- no unnecessary serving architecture.

---

## 22. Benchmark Methodology

Review:

- warm-up iterations,
- repeated measurements,
- consistent hardware,
- P50,
- P95,
- throughput,
- batch size,
- model-size calculation.

Flag single-run latency measurements presented as reliable benchmarks.

Do not require strict CI assertions for latency.

---

## 23. CPU / GPU Reporting

Verify no GPU metric is invented if no GPU is available.

Expected behavior:

```text
GPU benchmark not executed
```

is acceptable.

Likewise, ensure CPU comparisons are made on the same machine/environment where possible.

---

## 24. Model Size

Check whether model size is measured consistently.

For classical pipelines include both:

- vectorizer,
- classifier,

if both are needed for inference.

For Transformer include:

- model,
- tokenizer/config where material.

Do not compare only one component if deployment requires multiple artifacts.

---

## 25. Training Time

Check that timing covers the meaningful training operation.

For TF-IDF models, distinguish where useful:

```text
feature fit/transform
classifier fit
```

For Transformer, capture actual fine-tuning duration.

Do not compare unrelated timing scopes without noting the difference.

---

## 26. Explainability Comparison

Review claims such as:

```text
LR: high
SVM: high
Transformer: medium/low
```

Ensure the report explains why.

Do not accept attention weights alone as definitive Transformer explanations.

---

## 27. Cost / Operational Complexity

Review whether cost comparison is grounded.

Consider:

- CPU requirement,
- GPU requirement,
- memory,
- model artifact size,
- inference latency,
- throughput,
- deployment complexity,
- retraining complexity.

Do not invent API cost if the models are local.

---

## 28. Final Model Comparison

Verify that all three models are compared using the same final test set.

Required dimensions should include:

```text
Macro F1
Macro Precision
Macro Recall
Training Time
CPU latency
GPU latency when available
Model Size
Explainability
Calibration/probability behavior
Operational Complexity
Cost
```

The final recommendation must not automatically favor the Transformer.

---

## 29. Final Model Recommendation

Evaluate whether the recommendation reflects engineering trade-offs.

A small quality improvement may not justify a very large latency/cost increase.

Ask:

- Is the quality gain meaningful?
- Which classes improve?
- Is CPU deployment required?
- Are probabilities required?
- Is explainability important?
- Does the workload justify GPU serving?
- Is the complexity worth it?

The desired principle is:

> Choose the simplest model that meets the quality and operational requirements.

---

# Special Review: Over-Engineering

Explicitly inspect for unnecessary complexity.

Flag:

- generic model registry for only three models,
- large configuration framework,
- training framework wrappers hiding sklearn/HF behavior,
- experiment database/platform added prematurely,
- unnecessary service layer,
- unnecessary async model training,
- excessive generic abstractions,
- factory patterns with no real benefit.

For each abstraction ask:

> What current problem does this solve?

If the answer is only future-proofing, recommend simplification.

---

# Special Review: Under-Engineering

Also flag unreliable shortcuts.

Examples:

```text
training on full dataset
```

```text
TF-IDF fitted before split
```

```text
using test set to tune models
```

```text
accuracy-only evaluation
```

```text
no dataset version traceability
```

```text
no saved label mapping
```

```text
manual one-off notebook with no reusable pipeline
```

```text
one latency measurement
```

```text
SVM decision scores labeled as probabilities
```

```text
Transformer evaluated on a different split
```

The target is minimal clean ML engineering, not minimal code at any cost.

---

# Reproducibility Review

Verify that another developer can reproduce the final results from the repository.

Check:

- dependencies,
- dataset version,
- split seed,
- split manifest,
- model configuration,
- vectorizer configuration,
- tokenizer/model identifier,
- random seed,
- artifact locations,
- commands/instructions.

Flag undocumented manual steps that materially affect results.

---

# Data Safety Review

Verify:

- ingestion privacy guarantees remain intact,
- no raw PII is reintroduced,
- no sensitive ticket text is dumped unnecessarily into reports/logs,
- error-analysis artifacts are privacy conscious,
- original dataset remains untouched.

---

# Testing Review

Run/review the complete test suite.

At minimum expect strong deterministic tests for:

### Dataset/audit

- missing labels,
- class distribution,
- conflicting labels,
- leakage fields.

### Splitting

- determinism,
- no overlap,
- stable sizes,
- class behavior.

### TF-IDF

- train-only fitting,
- serialization,
- transform behavior.

### Classical models

- fit/predict,
- probability vs margin behavior,
- reload.

### Evaluation

- known metric examples,
- confusion matrix.

### Calibration

- deterministic calculation if implemented.

### Benchmark utilities

- mechanics only, not fixed machine-specific speed assertions.

### Transformer

- lightweight smoke coverage.

Long real-model training should not be mandatory in normal unit tests.

---

# Review Severity Levels

Classify findings as:

## BLOCKER

Must be fixed before the classification work can be considered complete.

Examples:

- train/test leakage,
- target leakage,
- broken final comparison,
- incorrect metrics,
- model evaluated on different test sets without justification,
- PII exposure,
- final test repeatedly used for tuning,
- failing core tests.

## MAJOR

Important correctness/design problem.

Examples:

- TF-IDF fitted on validation/test,
- SVM scores treated as probabilities,
- unreproducible split,
- wrong class mapping,
- misleading benchmarking,
- Transformer checkpoint chosen on test data.

## MINOR

Non-blocking improvement.

Examples:

- naming,
- small duplication,
- missing lightweight documentation,
- modest readability issue.

## OPTIONAL

Nice-to-have only.

Do not block approval for optional sophistication.

---

# Required Review Output

Return the review in this structure.

## 1. Verdict

Choose one:

```text
APPROVED
APPROVED WITH MINOR FIXES
CHANGES REQUIRED
```

Briefly explain.

---

## 2. Architecture Assessment

Evaluate:

- simplicity,
- modularity,
- reproducibility,
- debugability,
- maintainability,
- over-engineering risk.

---

## 3. Scope Coverage

Use a table:

| Requirement | Status | Evidence / Notes |
|---|---|---|
| Classification dataset audit | PASS/FAIL/PARTIAL | |
| Target/task definition | PASS/FAIL/PARTIAL | |
| Input field policy | PASS/FAIL/PARTIAL | |
| Leakage safety | PASS/FAIL/PARTIAL | |
| Reproducible split | PASS/FAIL/PARTIAL | |
| TF-IDF | PASS/FAIL/PARTIAL | |
| Logistic Regression | PASS/FAIL/PARTIAL | |
| Linear SVM | PASS/FAIL/PARTIAL | |
| Shared evaluation | PASS/FAIL/PARTIAL | |
| Error analysis | PASS/FAIL/PARTIAL | |
| Classical explainability | PASS/FAIL/PARTIAL | |
| Transformer tokenization | PASS/FAIL/PARTIAL | |
| Transformer fine-tuning | PASS/FAIL/PARTIAL | |
| Class imbalance analysis | PASS/FAIL/PARTIAL | |
| Threshold policy | PASS/FAIL/PARTIAL/N/A | |
| Calibration | PASS/FAIL/PARTIAL/N/A | |
| Batch inference | PASS/FAIL/PARTIAL | |
| CPU benchmark | PASS/FAIL/PARTIAL | |
| GPU benchmark | PASS/FAIL/PARTIAL/N/A | |
| Model size | PASS/FAIL/PARTIAL | |
| Training time | PASS/FAIL/PARTIAL | |
| Final comparison | PASS/FAIL/PARTIAL | |
| Model recommendation | PASS/FAIL/PARTIAL | |
| Tests | PASS/FAIL/PARTIAL | |
| Documentation | PASS/FAIL/PARTIAL | |

---

## 4. Scientific / Evaluation Integrity

Explicitly assess:

```text
split integrity
test-set discipline
feature leakage
target leakage
metric correctness
fairness of model comparison
```

Call out any result that cannot be trusted.

---

## 5. Findings

For every finding:

```text
Severity:
File:
Location:
Problem:
Why it matters:
Recommended fix:
```

Be concrete.

---

## 6. Model-by-Model Review

### Logistic Regression

State:

- correctness,
- strengths,
- issues.

### Linear SVM

State:

- correctness,
- strengths,
- issues.

### Transformer

State:

- correctness,
- strengths,
- issues.

---

## 7. Metrics Review

Verify the reported metrics and highlight any misleading use of:

```text
accuracy
macro averages
weighted averages
probabilities
thresholds
calibration metrics
```

---

## 8. Benchmark Review

Assess:

- timing methodology,
- hardware consistency,
- warmup,
- repetitions,
- P50/P95,
- throughput,
- GPU availability,
- model-size accounting.

---

## 9. Over-Engineering Review

Explicitly answer:

1. Is the implementation more complex than necessary?
2. Which abstractions should be removed or simplified?
3. Is any future-proofing premature?
4. Can a normal Python/ML developer debug the workflow easily?

---

## 10. Under-Engineering Review

Explicitly identify shortcuts that reduce trustworthiness.

---

## 11. Data Safety Review

Verify:

- Phase 1 privacy properties remain intact,
- no raw PII leakage,
- reports/logs are safe,
- original dataset remains unchanged.

---

## 12. Test Review

Report:

```text
test command
total tests
passed
failed
skipped
```

If tests cannot be executed, say so.

List only high-value missing tests.

---

## 13. Required Fixes

Provide a finite prioritized list of changes needed for approval.

Do not mix optional enhancements into this section.

---

## 14. Optional Improvements

Only practical improvements that materially help readability, maintainability, or experiment quality.

---

## 15. Final Model-Selection Review

Answer:

1. Is the recommended model actually supported by the evidence?
2. Is the gain over simpler models meaningful?
3. Were latency, size, explainability, and cost considered?
4. Would you choose the same model for production?

If not, state which model you would choose and why.

---

## 16. Final Completion Decision

Explicitly answer:

> Is this classification implementation complete and trustworthy enough to serve as the model-development foundation for the next system capability?

Answer:

```text
YES
```

or:

```text
NO
```

with a short reason.

---

# Reviewer Constraint

Do not rewrite the project merely to match your preferred ML architecture.

Review it against:

```text
correctness
data integrity
leakage safety
evaluation integrity
reproducibility
simplicity
industry practice
testability
debugability
maintainability
scope
```

A straightforward implementation with trustworthy experiments is better than an elaborate architecture.

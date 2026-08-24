# Phase 1 — Claude Code Review Prompt

## Role

Act as a **senior Python / NLP data-engineering reviewer**.

Review the Phase 1 implementation of the repository:

```text
support-ticket-ingestion-pipeline
```

The goal is to determine whether the implementation is:

- correct,
- simple,
- human-manageable,
- modular,
- reusable,
- easy to debug,
- aligned with industry coding practices,
- and appropriately scoped for a learning project.

This is **not** a request to redesign the project into a large enterprise architecture.

---

# Critical Review Principle

The implementation should follow:

> **The simplest design that cleanly solves the current Phase 1 requirements.**

Do not recommend complexity merely because it could be useful in a much larger production system.

Reject both extremes:

```text
throwaway tutorial code
```

and:

```text
unnecessary enterprise over-engineering
```

The desired result is:

```text
simple + modular + testable + debuggable + reusable
```

---

# Phase 1 Intended Scope

The implementation should cover:

1. Canonical ticket data contract.
2. CSV loading.
3. JSON loading.
4. Mapping supported sources into one canonical representation.
5. Schema validation.
6. Malformed-record detection.
7. Empty/effectively-empty message detection.
8. Unicode normalization.
9. Safe text normalization.
10. HTML/markup handling.
11. URL policy.
12. Email policy.
13. Emoji policy.
14. Punctuation policy.
15. PII masking.
16. Exact duplicate detection.
17. Lightweight semantic duplicate support only if justified.
18. Language detection.
19. Data-leakage field checks.
20. Dataset/pipeline version metadata.
21. Data-quality reporting.
22. Accepted-record output.
23. Rejected-record output with reasons.
24. Automated unit tests.
25. One end-to-end integration test.

---

# Explicitly Out of Scope

Flag unnecessary implementation of:

- intent classification,
- priority prediction models,
- Naive Bayes,
- Logistic Regression,
- SVM,
- transformer classifiers,
- model train/validation/test splitting,
- RAG,
- vector databases,
- LLM generation,
- fine-tuning,
- agent systems,
- complex service infrastructure,
- microservices,
- queues,
- unnecessary databases.

If any of these were added without a clear Phase 1 requirement, classify them as over-engineering.

---

# Dataset Conventions

Expected naming:

```text
tickets_original.csv
```

= immutable original source dataset.

```text
tickets_raw.csv
```

= working Phase 1 input.

The implementation should not modify `tickets_original.csv`.

JSON support does not require duplicating the entire production dataset into JSON; a small input or fixture is sufficient to prove the loader works.

---

# Review Areas

Review each area separately.

---

## 1. Repository Structure

Check whether the repository is organized clearly.

Expected responsibilities may resemble:

```text
models.py
loaders.py
validators.py
normalization.py
pii.py
deduplication.py
language.py
leakage.py
versioning.py
quality.py
pipeline.py
```

Do not require this exact file structure if the existing structure is equally clear.

Evaluate:

- Is each module cohesive?
- Are responsibilities obvious?
- Is code unnecessarily fragmented?
- Is too much logic concentrated in one giant module?
- Are modules genuinely reusable?
- Can a developer quickly find the relevant code?

Flag both:

```text
god modules
```

and:

```text
one-function-per-file over-fragmentation
```

---

## 2. Simplicity and Human Debuggability

Check whether a developer can trace a single ticket through the system.

Ask:

- Is control flow explicit?
- Are function names descriptive?
- Are inputs/outputs understandable?
- Are data transformations visible?
- Are errors understandable?
- Are there unnecessary wrappers or indirection?
- Are abstractions solving actual duplication/problems?

Flag unnecessary:

- factories,
- registries,
- base-class hierarchies,
- dependency-injection systems,
- plugin architectures,
- generic frameworks,
- excessive decorators,
- complex async logic.

Recommend simpler alternatives where appropriate.

---

## 3. Canonical Data Contract

Review the Pydantic ticket model.

Check:

- actual source data informed the schema,
- required fields are reasonable,
- optional/null fields are intentional,
- types are appropriate,
- categorical constraints are not overly restrictive,
- source-specific naming is mapped cleanly,
- the canonical model is not polluted with unnecessary implementation details.

Verify the implementation understands:

```text
schema-valid
≠
business-valid
```

A record such as:

```text
message = "     "
```

may satisfy a string type but should still be rejected as unusable.

---

## 4. CSV / JSON Loading

Check:

- both formats are supported,
- both feed one common downstream pipeline,
- source-specific parsing is isolated,
- duplicate processing logic is avoided,
- malformed file-level errors are clear,
- unsupported formats fail explicitly.

Do not accept separate full pipelines for CSV and JSON.

---

## 5. Validation

Review how invalid records are handled.

Expected behavior:

```text
bad individual record
→ reject record with reason
→ continue processing other records
```

while:

```text
missing input file / unreadable source
→ fail pipeline clearly
```

Check:

- no silent row dropping,
- rejection reasons are understandable,
- validation taxonomy is not unnecessarily complex,
- errors preserve enough context for debugging.

---

## 6. Empty / Malformed Message Detection

Verify cases such as:

```text
None
""
"     "
"\n\t"
"<p></p>"
```

are handled appropriately.

Check whether effective emptiness is evaluated after necessary lightweight markup handling.

Avoid excessively clever heuristics.

---

## 7. Unicode Normalization

Check that Unicode normalization is:

- deterministic,
- documented,
- implemented before comparisons where appropriate.

A conservative form such as NFC is acceptable unless another form is justified.

Ensure normalization handles equivalent Unicode representations consistently.

---

## 8. Text Normalization Policy

Review whether cleaning preserves useful language.

Flag destructive behavior such as:

```python
re.sub(r"[^\w\s]", "", text)
```

when applied globally.

The pipeline should generally preserve:

```text
!
?
apostrophes
technical punctuation
emoji
case
```

unless there is a documented reason to transform them.

Verify the implementation does **not** automatically:

- lowercase everything,
- remove stop words,
- stem all words,
- lemmatize all words,
- remove all punctuation.

Those transformations are not required for the canonical Phase 1 dataset.

---

## 9. HTML / Markup Handling

Verify:

```text
<p>Hello <b>team</b></p>
```

becomes readable text such as:

```text
Hello team
```

without accidental word concatenation or excessive whitespace.

Check malformed HTML behavior.

---

## 10. URL / Email / Emoji / Punctuation Policies

Review whether each has an explicit, consistent policy.

### Email

Email is PII and should be masked.

### URL

A simple replacement such as:

```text
<URL>
```

is acceptable if documented.

### Emoji

Should generally be preserved.

### Punctuation

Meaningful punctuation should generally be preserved.

Confirm technical strings are not unintentionally damaged:

```text
E-102
v2.4.1
/api/payment
C++
```

---

## 11. PII Masking

Review masking for:

```text
email
phone
IP address
```

Check:

- placeholders are consistent,
- original values are not exposed in processed output,
- masking does not destroy surrounding useful text,
- false positives are reasonably controlled,
- raw PII is not unnecessarily written to logs,
- counts can feed the quality report.

Do not require an enterprise DLP engine.

---

## 12. Exact Duplicate Detection

Verify duplicate detection is deterministic.

Check:

- which message representation is used,
- normalization happens consistently,
- stable comparison/hash is used if appropriate,
- duplicates are traceable,
- duplicate records are not silently discarded.

If `duplicate_of` metadata is used, verify it is correct.

Look for accidental order-dependent behavior.

---

## 13. Semantic Duplicate Detection

This feature must remain lightweight.

If implemented, check:

- it is isolated from core exact deduplication,
- it does not require a complex vector infrastructure,
- threshold/configuration is understandable,
- it does not dominate the Phase 1 architecture.

If the implementation became complex because of semantic duplicates, recommend simplifying or deferring it.

---

## 14. Language Detection

Check:

- language is stored as metadata,
- uncertain results are handled,
- non-English tickets are not automatically rejected without a documented reason,
- the chosen library/approach is proportionate to the project.

---

## 15. Data Leakage Checks

Review whether source fields were inspected for future prediction-time availability.

Potential leakage fields may include:

```text
resolution
agent_response
resolved_at
final_status
post-resolution notes
```

The exact list must be based on the actual dataset.

Check whether the implementation documents reasoning for categories such as:

```text
SAFE_FOR_FUTURE_MODEL_INPUT
POTENTIAL_LEAKAGE
METADATA_ONLY
```

No model training is required.

---

## 16. Dataset Versioning

Verify the system records enough information to identify the processed dataset.

Accept simple metadata such as:

```text
dataset version
source file
source row count
normalization version
source checksum
```

Do not require a dedicated dataset-versioning platform.

---

## 17. Data-Quality Report

Review report correctness.

Expected metrics may include:

```text
total_records
accepted_records
rejected_records
rejection_rate
empty_messages
malformed_records
exact_duplicates
semantic_duplicate_candidates
emails_masked
phones_masked
ip_addresses_masked
language_counts
```

Check:

- metrics match actual outputs,
- percentages use correct denominators,
- counts are not double-counted,
- report generation is separated enough to test,
- missing optional metrics are handled cleanly.

---

## 18. Accepted / Rejected Outputs

Accepted output should be model-ready and privacy-safe.

Rejected output should contain enough information to debug why a record failed.

Check that records are not silently lost.

Expected examples:

```text
tickets_clean.jsonl
tickets_rejected.jsonl
data_quality_report.json
```

Do not reject the implementation merely because CSV is used instead of JSONL if the choice is documented and consistent; evaluate the design rather than file-extension preference.

---

## 19. Pipeline Orchestration

Review the main pipeline flow.

Expected shape:

```text
load
 ↓
canonical mapping
 ↓
schema validation
 ↓
business validation
 ↓
normalize
 ↓
mask PII
 ↓
deduplicate
 ↓
language detection
 ↓
quality metrics
 ↓
write outputs
```

Check ordering carefully.

Look for issues such as:

- deduplication before appropriate normalization,
- PII leakage into processed output,
- metrics counted before/after transformations inconsistently,
- invalid records reaching downstream logic.

`pipeline.py` should coordinate modules rather than contain every implementation detail.

---

## 20. Error Handling

Flag:

```python
except Exception:
    pass
```

and similar swallowed errors.

Review distinction between:

### Run-level/system error

Examples:

```text
file missing
invalid source document
permission failure
unwritable output
```

These should fail clearly.

### Record-level data error

Examples:

```text
invalid timestamp
empty message
invalid field value
```

These should normally reject the record and allow the pipeline to continue.

---

## 21. Logging

Check logging is:

- useful,
- concise,
- free from unnecessary raw PII,
- not excessively verbose,
- helpful for tracing counts/stages.

Good examples:

```text
Loaded N records
Rejected N during validation
Detected N exact duplicates
Wrote N accepted records
```

---

## 22. Tests

Run/review the full test suite.

At minimum expect tests for:

### Loader

- CSV,
- JSON,
- broken/unsupported source.

### Validation

- valid record,
- required missing field,
- invalid type,
- invalid timestamp,
- business-invalid message.

### Normalization

- Unicode,
- HTML,
- whitespace,
- punctuation preservation,
- emoji preservation.

### PII

- email,
- phone,
- IP.

### Deduplication

- exact duplicate,
- obvious non-duplicate.

### Quality report

- exact expected counts.

### Integration

One controlled fixture should run through the whole pipeline.

Verify tests assert behavior rather than implementation details.

Flag brittle tests.

---

# Special Review: Over-Engineering

Explicitly inspect for unnecessary complexity.

For every abstraction, ask:

> What current problem does this solve?

Flag it if the answer is only:

> We may need this later.

Examples:

- abstract base loader with only two simple loaders,
- complex dependency injection,
- excessive protocols/interfaces,
- factories for a fixed set of functions,
- multiple layers wrapping simple regex functions,
- unnecessary async processing,
- hidden global state,
- generic event systems,
- database-backed tracking for a local pipeline,
- elaborate configuration inheritance.

Suggest the simplest maintainable alternative.

---

# Special Review: Under-Engineering

Also identify code that is too simplistic for reliable industry practice.

Examples:

```python
df.dropna()
```

with no rejection reporting.

```python
text.lower().replace(...)
```

with destructive preprocessing and no policy.

Manually checking output instead of tests.

One giant script containing all behavior.

Hard-coded absolute paths.

Silent duplicate deletion.

PII left in processed output.

No distinction between bad rows and failed pipeline execution.

The target is not minimal code at any cost; it is **minimal clean architecture**.

---

# Review Severity Levels

Classify findings as:

## BLOCKER

Must be fixed before Phase 1 can be considered complete.

Examples:

- data loss,
- PII leakage,
- major incorrect validation,
- broken pipeline,
- failed core tests,
- unusable outputs,
- duplicate metrics fundamentally wrong.

## MAJOR

Important design/correctness problem.

Examples:

- unclear module boundaries,
- brittle error handling,
- destructive text normalization,
- incorrect processing order,
- inadequate rejection traceability.

## MINOR

Improvement that does not invalidate the implementation.

Examples:

- naming,
- small duplication,
- missing type hint,
- weak docstring,
- readability improvement.

## OPTIONAL

Nice-to-have improvement that should **not** block approval.

---

# Required Review Output

Return the review in the following structure.

## 1. Verdict

Choose one:

```text
APPROVED
APPROVED WITH MINOR FIXES
CHANGES REQUIRED
```

Explain the verdict briefly.

---

## 2. Architecture Assessment

Evaluate:

- simplicity,
- modularity,
- reusability,
- debugability,
- over-engineering risk,
- maintainability.

---

## 3. Phase 1 Scope Coverage

Use a table:

| Requirement | Status | Evidence / Notes |
|---|---|---|
| Canonical schema | PASS/FAIL/PARTIAL | ... |
| CSV loader | PASS/FAIL/PARTIAL | ... |
| JSON loader | PASS/FAIL/PARTIAL | ... |
| Schema validation | PASS/FAIL/PARTIAL | ... |
| Empty/malformed detection | PASS/FAIL/PARTIAL | ... |
| Unicode normalization | PASS/FAIL/PARTIAL | ... |
| HTML handling | PASS/FAIL/PARTIAL | ... |
| URL/email/emoji/punctuation policy | PASS/FAIL/PARTIAL | ... |
| PII masking | PASS/FAIL/PARTIAL | ... |
| Exact duplicates | PASS/FAIL/PARTIAL | ... |
| Semantic duplicates | PASS/FAIL/PARTIAL/N/A | ... |
| Language detection | PASS/FAIL/PARTIAL | ... |
| Leakage checks | PASS/FAIL/PARTIAL | ... |
| Dataset versioning | PASS/FAIL/PARTIAL | ... |
| Quality report | PASS/FAIL/PARTIAL | ... |
| Accepted/rejected outputs | PASS/FAIL/PARTIAL | ... |
| Unit tests | PASS/FAIL/PARTIAL | ... |
| Integration test | PASS/FAIL/PARTIAL | ... |

---

## 4. Findings

For every finding provide:

```text
Severity:
File:
Location:
Problem:
Why it matters:
Recommended fix:
```

Be specific.

Do not give vague advice such as:

> improve error handling.

State the exact issue.

---

## 5. Over-Engineering Review

Explicitly answer:

1. Is the implementation more complex than necessary?
2. Which abstractions should be removed or simplified?
3. Is any future-proofing premature?
4. Can a normal Python developer debug this code easily?

---

## 6. Reuse / Modularity Review

Identify:

- useful reusable functions,
- duplicated logic,
- overly coupled modules,
- functions that should be extracted,
- functions split unnecessarily.

---

## 7. Data Safety Review

Explicitly verify:

- original dataset remains untouched,
- PII does not leak into cleaned output,
- bad rows are traceable,
- duplicate handling does not silently lose data,
- raw PII is not unnecessarily logged.

---

## 8. Test Review

Report:

```text
test command
total tests
passed
failed
skipped
```

If you cannot execute tests, say so clearly.

List missing high-value test cases.

---

## 9. Required Fixes

Provide only fixes necessary for approval.

Keep this list prioritized and finite.

Do not mix optional refactoring with required correctness fixes.

---

## 10. Optional Improvements

List only improvements that would materially help readability or maintainability without increasing complexity.

---

## 11. Final Phase 1 Completion Decision

Explicitly answer:

> Is this Phase 1 implementation ready to be considered complete and used as the cleaned-data foundation for Phase 2?

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

Do not rewrite the project merely to match your preferred architecture.

Review the implementation against:

```text
correctness
clarity
simplicity
industry practice
testability
reusability
debugability
Phase 1 scope
```

A straightforward solution that meets these criteria is preferable to a sophisticated one.

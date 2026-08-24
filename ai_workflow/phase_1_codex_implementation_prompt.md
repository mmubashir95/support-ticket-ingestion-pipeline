# Phase 1 — Codex Implementation Prompt

## Project

**Repository:** `support-ticket-ingestion-pipeline`

## Objective

Implement **Phase 1 — Text Processing Foundations / Industry Text Data Engineering** as a production-style support-ticket ingestion pipeline.

The goal is to turn unreliable raw support-ticket data into:

- validated data,
- normalized text,
- privacy-safe text,
- duplicate-aware records,
- accepted/rejected outputs,
- a measurable data-quality report,
- and automated tests.

This implementation is for **industry practice**, but the code must remain easy for a human developer to understand, debug, maintain, and extend.

---

# Core Engineering Rules

These rules are mandatory.

## 1. Keep the code simple

Prefer straightforward Python over clever abstractions.

Do not introduce patterns, frameworks, base classes, factories, registries, plugin systems, dependency-injection containers, or other abstractions unless there is a clear current need.

A developer should be able to open a module and quickly understand:

- what it does,
- what inputs it accepts,
- what it returns,
- and where to debug a problem.

## 2. Do not over-engineer

Build only what Phase 1 needs.

Do **not** implement architecture for hypothetical future requirements.

Avoid:

- unnecessary class hierarchies,
- generic framework-style code,
- excessive configuration layers,
- excessive indirection,
- premature optimization,
- complex asynchronous processing,
- databases,
- queues,
- microservices,
- Docker unless specifically required later.

## 3. Use modular code

Separate responsibilities into small, meaningful modules.

Modules should be reusable but not fragmented into dozens of tiny files.

Prefer cohesive functions with clear names.

## 4. Make debugging easy

Prefer:

- explicit control flow,
- clear validation errors,
- meaningful rejection reasons,
- predictable functions,
- simple data structures,
- structured logging where useful.

Do not hide errors behind broad exception handling.

Never use:

```python
except Exception:
    pass
```

If an error is handled, preserve enough context to understand what failed.

## 5. Reuse code

Do not duplicate parsing, validation, normalization, masking, or reporting logic.

Create reusable functions for behavior that is genuinely shared.

## 6. Follow standard Python practices

Use:

- type hints,
- descriptive names,
- docstrings where they add value,
- small cohesive functions,
- Pydantic for schema validation,
- pytest for tests,
- `pathlib` for paths,
- Python standard library where sufficient.

Prefer readable code over compressed code.

---

# Phase 1 Scope

Implement the following capabilities:

1. Define the canonical support-ticket data contract.
2. Load CSV records.
3. Load JSON records.
4. Convert supported inputs into one canonical internal representation.
5. Validate schema and required fields.
6. Detect malformed records.
7. Detect empty or effectively empty messages.
8. Normalize Unicode.
9. Normalize text safely.
10. Remove HTML/markup without destroying useful text.
11. Define and implement URL, email, emoji, and punctuation handling policies.
12. Mask personally identifiable information.
13. Detect exact duplicate tickets.
14. Support a simple, clearly isolated semantic-duplicate capability only if it can be implemented without making the project unnecessarily complex.
15. Detect ticket language.
16. Identify obvious data-leakage-risk fields for future modeling.
17. Track dataset/pipeline version metadata.
18. Produce a data-quality report.
19. Save accepted records.
20. Save rejected records with clear rejection reasons.
21. Add automated unit tests.
22. Add one end-to-end integration test.

---

# Explicitly Out of Scope

Do **not** implement:

- intent classification,
- priority prediction models,
- Naive Bayes,
- Logistic Regression,
- SVM,
- transformer classifiers,
- train/validation/test model splitting,
- RAG,
- vector databases,
- chatbot functionality,
- LLM summarization,
- model fine-tuning,
- model-serving APIs,
- agent systems.

Phase 1 ends when reliable, model-ready ticket data is produced.

---

# Dataset Naming

Use these names:

```text
tickets_original.csv
```

Meaning:

> Immutable copy of the original downloaded dataset.

And:

```text
tickets_raw.csv
```

Meaning:

> Working input used by the Phase 1 ingestion pipeline.

Do not modify `tickets_original.csv`.

The pipeline should process `tickets_raw.csv`.

JSON support can be demonstrated using a small JSON input or test fixture. The main dataset does not need to be duplicated entirely into JSON.

---

# Recommended Repository Structure

Use a structure close to the following:

```text
support-ticket-ingestion-pipeline/
│
├── data/
│   ├── source/
│   │   └── tickets_original.csv
│   │
│   ├── raw/
│   │   └── tickets_raw.csv
│   │
│   ├── processed/
│   │   └── tickets_clean.jsonl
│   │
│   └── rejected/
│       └── tickets_rejected.jsonl
│
├── reports/
│   └── data_quality_report.json
│
├── src/
│   └── ticket_pipeline/
│       ├── __init__.py
│       ├── models.py
│       ├── loaders.py
│       ├── validators.py
│       ├── normalization.py
│       ├── pii.py
│       ├── deduplication.py
│       ├── language.py
│       ├── leakage.py
│       ├── versioning.py
│       ├── quality.py
│       └── pipeline.py
│
├── scripts/
│   └── run_pipeline.py
│
├── tests/
│   ├── fixtures/
│   │   ├── tickets_valid.csv
│   │   ├── tickets_invalid.csv
│   │   └── tickets_sample.json
│   │
│   ├── test_loaders.py
│   ├── test_validation.py
│   ├── test_normalization.py
│   ├── test_pii.py
│   ├── test_deduplication.py
│   ├── test_quality.py
│   └── test_pipeline_integration.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

You may adjust this structure if the existing repository already has a clean equivalent.

Do not add modules merely to match the diagram if they are unnecessary.

---

# Implementation Order

Implement the work in the following order.

---

## Phase 1.1 — Canonical Ticket Data Contract

Create a Pydantic model representing the ticket shape used internally by the pipeline.

Before defining the model:

1. Inspect the actual dataset columns.
2. Identify which fields exist.
3. Map source column names into the canonical schema.
4. Avoid inventing fields that are not present unless needed as generated metadata.

The contract should distinguish:

- required fields,
- optional fields,
- nullable fields,
- expected types,
- allowed categorical values where appropriate.

Examples of candidate canonical fields may include:

```text
ticket_id
created_at
subject
message
channel
priority
```

but use the actual dataset as the authority.

### Important distinction

Schema-valid means:

> The record has the expected technical structure and types.

Business-valid means:

> The record is actually usable for the support/NLP task.

For example:

```text
message = "     "
```

may be a valid string but is not a usable support message.

Keep these concerns separate in the code.

---

## Phase 1.2 — CSV and JSON Loaders

Implement simple loaders for:

- CSV
- JSON

Both loaders must return records that can be mapped into the same canonical representation.

Do not build separate cleaning pipelines for CSV and JSON.

Desired architecture:

```text
CSV  ──┐
       ├──> canonical record representation
JSON ──┘
                ↓
          common pipeline
```

Requirements:

- clear error when file cannot be read,
- clear error for unsupported input format,
- no silent failures,
- keep loader functions simple.

---

## Phase 1.3 — Schema Validation

Validate each loaded record against the canonical Pydantic model.

Do not terminate the entire pipeline because one record is invalid.

Separate records into:

```text
accepted-for-further-processing
rejected
```

Rejected records must include a machine-readable reason.

Examples:

```text
MISSING_REQUIRED_FIELD
INVALID_TIMESTAMP
INVALID_FIELD_TYPE
INVALID_CHANNEL
```

Use a small, understandable set of rejection codes.

Do not create an elaborate error taxonomy.

---

## Phase 1.4 — Empty and Malformed Message Detection

Detect messages such as:

```text
None
""
"     "
"\n\t"
"<p></p>"
```

A record can pass schema validation but still fail business validation.

After lightweight HTML/text interpretation, reject records with no meaningful message content.

Use a clear reason such as:

```text
EMPTY_MESSAGE
```

Avoid heuristic complexity.

---

## Phase 1.5 — Deterministic Text Normalization

Implement deterministic normalization.

Same input must always produce the same output.

At minimum cover:

### Unicode normalization

Use a clearly documented normalization form.

Prefer a conservative default such as:

```python
unicodedata.normalize("NFC", text)
```

unless the dataset provides a clear reason to use another form.

### HTML / markup

Remove markup while preserving readable content.

Example:

```text
<p>Hello <b>team</b></p>
```

should become roughly:

```text
Hello team
```

### Whitespace

Normalize unnecessary repeated whitespace without destroying useful sentence structure.

### Control characters

Remove or normalize problematic control characters when appropriate.

### Newlines

Apply a documented policy.

---

# Text Preservation Policy

Do not aggressively clean natural language.

Do **not** globally:

```text
lowercase everything
remove all punctuation
remove stop words
stem every word
lemmatize every word
```

Those transformations are not required for the Phase 1 canonical cleaned dataset.

Preserve natural text unless there is a clear reason to change it.

---

## Phase 1.6 — URL, Email, Emoji, and Punctuation Policy

Make the behavior explicit.

### Punctuation

Keep meaningful punctuation by default.

Examples:

```text
Payment failed!!!
Why was I charged?!
don't
E-102
v2.4.1
/api/payment
C++
```

Do not use a destructive regex such as:

```python
re.sub(r"[^\w\s]", "", text)
```

as the general cleaning strategy.

### Emoji

Keep emoji by default because it may contain sentiment or intent information.

Example:

```text
This is terrible 😡
```

should not automatically lose the emoji.

### Email

Email addresses are PII.

Detect and mask them.

### URL

Choose and document a simple policy.

A reasonable Phase 1 policy is to replace full URLs with:

```text
<URL>
```

if preserving the literal URL is not needed.

Do not damage technical text before deciding whether it is a URL, version, path, or error code.

---

## Phase 1.7 — PII Masking

Implement simple, explicit PII masking.

Start with:

```text
email
phone number
IP address
```

Use placeholders such as:

```text
<EMAIL>
<PHONE>
<IP>
```

Example:

```text
Contact me at ali@example.com or 03001234567
```

becomes:

```text
Contact me at <EMAIL> or <PHONE>
```

Do not remove the entire phrase because the fact that an email/phone was provided may still carry useful structural information.

Track counts of masked PII for the quality report.

Do not attempt a complete enterprise DLP system.

---

## Phase 1.8 — Exact Duplicate Detection

Implement exact duplicate detection using a deterministic canonical representation of the message.

A reasonable approach:

```text
normalized message
      ↓
hash
      ↓
duplicate lookup
```

Use a stable hash such as SHA-256 if hashing is helpful.

Preserve traceability.

A duplicate record should be able to identify its original/canonical record when practical.

Example metadata:

```json
{
  "duplicate_of": "TKT-100"
}
```

Do not silently delete duplicates without recording what happened.

---

## Phase 1.9 — Semantic Duplicate Detection

Keep this capability deliberately lightweight.

The goal is to distinguish:

```text
exact duplicate
near duplicate
semantic duplicate
```

Do not introduce a vector database or production retrieval system.

If semantic duplicate detection requires substantial dependencies or architecture, isolate it behind a small optional function/module and keep it disabled by default.

The core Phase 1 pipeline must remain understandable without it.

---

## Phase 1.10 — Language Detection

Detect the likely language of usable ticket messages.

Store language as metadata.

Do not automatically reject non-English messages unless a documented business rule requires it.

Possible metadata:

```text
language = "en"
language = "fr"
language = "unknown"
```

Keep the implementation simple.

---

## Phase 1.11 — Data Leakage Checks

This is preparation for future model training.

Inspect source columns and identify fields that may contain information unavailable at ticket-creation/prediction time.

Examples may include:

```text
resolution
agent_response
resolved_at
final_status
post-resolution notes
```

Do not train a model.

Create a small, understandable mechanism to classify fields as:

```text
SAFE_FOR_FUTURE_MODEL_INPUT
POTENTIAL_LEAKAGE
METADATA_ONLY
```

Base the classification on the actual dataset.

Document the reasoning.

---

## Phase 1.12 — Dataset Versioning

Record enough metadata to reproduce which dataset and pipeline configuration produced an output.

Keep this simple.

Example:

```json
{
  "dataset_version": "1.0",
  "source_file": "tickets_raw.csv",
  "source_row_count": 61800,
  "normalization_version": "1.0"
}
```

Add a source checksum if straightforward.

Do not introduce a dedicated versioning platform.

---

## Phase 1.13 — Data-Quality Report

Produce a JSON report.

Include metrics such as:

```text
total_records
accepted_records
rejected_records
empty_messages
malformed_records
exact_duplicates
semantic_duplicate_candidates
emails_masked
phones_masked
ip_addresses_masked
language_counts
```

Include percentages where useful.

Example structure:

```json
{
  "total_records": 10000,
  "accepted_records": 9412,
  "rejected_records": 588,
  "rejection_rate": 0.0588,
  "empty_messages": 142,
  "exact_duplicates": 253,
  "pii": {
    "emails_masked": 1945,
    "phones_masked": 672
  }
}
```

Keep metric computation transparent and testable.

---

## Phase 1.14 — Pipeline Orchestration

Create one straightforward orchestrator.

Conceptual flow:

```text
load
 ↓
map to canonical structure
 ↓
schema validation
 ↓
business validation
 ↓
normalize
 ↓
mask PII
 ↓
detect duplicates
 ↓
detect language
 ↓
collect quality metrics
 ↓
save accepted/rejected/report
```

The orchestrator should call reusable modules.

Do not put all logic into `pipeline.py`.

Likewise, do not turn every function into a class.

Prefer simple function composition.

---

## Phase 1.15 — Outputs

Produce:

```text
data/processed/tickets_clean.jsonl
data/rejected/tickets_rejected.jsonl
reports/data_quality_report.json
```

Accepted records should contain cleaned, model-ready text and relevant metadata.

Rejected records must preserve enough source context to debug the rejection.

Example:

```json
{
  "ticket_id": "TKT-20",
  "rejection_reason": "EMPTY_MESSAGE"
}
```

Do not silently discard rejected rows.

---

## Phase 1.16 — Automated Unit Tests

Use pytest.

At minimum test:

### Loaders

- valid CSV loads,
- valid JSON loads,
- unsupported/broken input fails clearly.

### Validation

- valid record passes,
- required field missing,
- invalid timestamp,
- invalid type,
- empty message business validation.

### Normalization

- Unicode normalization,
- HTML removal,
- whitespace handling,
- meaningful punctuation preserved,
- emoji preserved.

### PII

- email masking,
- phone masking,
- IP masking,
- ordinary text not incorrectly destroyed.

### Duplicates

- exact duplicate detected,
- non-duplicate not incorrectly marked.

### Quality report

- counts are correct.

Tests should be deterministic.

---

## Phase 1.17 — End-to-End Integration Test

Create one small controlled fixture containing examples such as:

```text
valid ticket
HTML ticket
email PII
phone PII
empty message
invalid timestamp
duplicate ticket
Unicode variation
emoji
```

Run the entire pipeline.

Assert:

- expected accepted count,
- expected rejected count,
- PII is masked,
- duplicates are identified,
- normalization occurred,
- output files are generated,
- quality report values match actual outputs.

---

# Logging

Use Python logging where it helps trace pipeline execution.

Keep logging concise.

Good examples:

```text
Loaded 61,800 records from tickets_raw.csv
Rejected 143 records during schema validation
Detected 927 exact duplicates
Wrote 60,213 accepted records
```

Do not log raw PII unnecessarily.

---

# Configuration

Keep configuration minimal.

A small configuration object/file is acceptable for things such as:

```text
input path
output paths
normalization version
semantic duplicate threshold
```

Do not create a large configuration framework.

---

# Error Handling

Prefer clear failures for system-level problems:

```text
input file missing
invalid JSON document
output directory cannot be written
unsupported format
```

Prefer record-level rejection for bad individual records.

The distinction should be:

```text
System problem
→ fail the run clearly

Individual bad ticket
→ reject ticket and continue
```

---

# Code Quality Expectations

The final implementation should feel like code a small professional engineering team could maintain.

A reviewer should be able to:

1. trace one ticket through the pipeline,
2. find where validation happens,
3. find where normalization happens,
4. find where PII is masked,
5. inspect why a record was rejected,
6. run tests easily,
7. change one policy without rewriting the project.

---

# README Requirements

Update the README with:

1. project purpose,
2. Phase 1 scope,
3. repository structure,
4. setup instructions,
5. how to run the pipeline,
6. how to run tests,
7. input/output files,
8. text-normalization policy,
9. PII policy,
10. duplicate policy,
11. known limitations,
12. explicit list of Phase 1 out-of-scope features.

Keep the README practical rather than overly long.

---

# Implementation Workflow

Before editing code:

1. inspect the current repository,
2. inspect the dataset columns,
3. inspect existing dependencies,
4. reuse existing code where appropriate,
5. identify the smallest clean implementation.

Then implement incrementally.

After each meaningful step:

```bash
pytest
```

Run the full test suite at the end.

---

# Final Codex Response

At completion, provide:

## 1. Summary

Briefly explain what was implemented.

## 2. Files Changed

List created/modified files and their purpose.

## 3. Pipeline Flow

Show the final processing order.

## 4. Validation/Rejection Rules

List the implemented rules.

## 5. Text Policy

State the implemented behavior for:

- Unicode,
- HTML,
- whitespace,
- punctuation,
- emoji,
- URLs,
- emails.

## 6. PII

State which PII types are masked.

## 7. Deduplication

Explain exact and, if implemented, semantic duplicate handling.

## 8. Tests

Report:

- test command,
- total tests,
- pass/fail result.

## 9. Outputs

State which files the pipeline generates.

## 10. Limitations / Deferred Work

Clearly identify anything intentionally not implemented.

Do not claim work is complete if tests fail.

# Phase 2.2 — Small-Step Study + Implementation Plan

Phase 2.2 should be completed in small checkpoints instead of implementing the whole train / validation / test split at once.

The goal of Phase 2.2 is to create **one reproducible train / validation / test split that every later model will reuse**, with stratification where practical and with split metadata stored for reproducibility.

---

## Step 2.2.1 — Understand Why We Split the Dataset

### Study first

Understand the purpose of:

```text
Train set
Validation set
Test set
```

You should be able to explain:

```text
Train
→ model learns from this

Validation
→ used during development/tuning

Test
→ untouched until final evaluation
```

For this project, use:

```text
70% Train
15% Validation
15% Test
```

The exact percentages are less important than keeping them reproducible.

### Important concept

Understand why this is wrong:

```text
Entire dataset
↓
Fit TF-IDF
↓
Split
```

And why this is correct:

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

This prevents information leakage from validation/test into training.

### No coding yet

At this step, just understand:

> Why do train, validation, and test need to exist separately?

### Exit criteria

You should be able to answer:

- What is train data?
- What is validation data?
- What is test data?
- Why shouldn't the test set influence development?
- Why must splitting happen before TF-IDF?

---

## Step 2.2.2 — Understand Stratified Splitting

Current classes:

```text
Incident   11,466
Request     8,187
Problem     6,012
Change      2,922
```

The dataset has mild imbalance but no rare classes.

### Study

Learn what **stratification** means.

Instead of randomly splitting without considering classes:

```text
Dataset
↓
random split
```

we try to preserve approximately the same class percentages in:

```text
Train
Validation
Test
```

For example:

```text
Full dataset

Incident ≈ 40%
Request  ≈ 29%
Problem  ≈ 21%
Change   ≈ 10%
```

Then ideally:

```text
Train
Incident ≈ 40%
Request  ≈ 29%
Problem  ≈ 21%
Change   ≈ 10%

Validation
similar distribution

Test
similar distribution
```

### Exit criteria

You should understand:

> Stratification preserves label distribution across splits.

---

## Step 2.2.3 — Freeze the Split Configuration

Before splitting actual records, define the configuration.

Example:

```text
dataset_version = ds_6fa2f19cf683

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

random_seed = 42

target = ticket_type

stratified = true
```

The audit has already frozen:

```text
target = ticket_type
input = subject + message
task = single_label_multiclass
```

### Why this step exists

We want the split behavior to be explicit rather than buried inside code.

Later, if you run:

```text
same dataset
+
same configuration
+
same seed
```

you should get:

```text
same split
```

### Implementation

Create only the configuration/model structures.

For example conceptually:

```text
SplitConfig
```

with:

```text
train_ratio
validation_ratio
test_ratio
random_seed
stratify
dataset_version
```

Do **not** perform splitting yet.

### Tests

Test:

```text
70 + 15 + 15 = 100%
```

Reject things like:

```text
70 + 20 + 20
```

or:

```text
negative ratio
```

### Exit criteria

You have a valid, frozen split configuration.

---

## Step 2.2.4 — Implement the Basic Deterministic Split

Now perform the first actual split.

Input:

```text
ClassificationRecord[]
```

Output conceptually:

```text
SplitResult

train
validation
test
```

At this stage focus only on:

```text
same seed → same records
```

### Important concept

Understand **random seed**.

Without fixed seed:

```text
Run 1
Train = A B C D

Run 2
Train = A C E F
```

With fixed seed:

```text
Run 1 with seed 42
→ same split

Run 2 with seed 42
→ same split
```

### Tests

Verify:

```text
same seed
→ same IDs
```

and:

```text
different split groups
→ no overlapping IDs
```

### Exit criteria

You have a reproducible split.

---

## Step 2.2.5 — Add Stratification

Now upgrade the deterministic split so class balance is preserved.

This is where you use:

```text
ticket_type
```

for stratification.

Conceptually:

```text
ClassificationRecord
    ↓
group by label
    ↓
split each label proportionally
    ↓
combine
```

You don't necessarily need to implement the algorithm manually if you use Scikit-learn, but you should understand the concept.

### Check the result

Create a comparison like:

```text
                 Full    Train    Validation   Test

Incident         40.1%    ~40%       ~40%      ~40%
Request          28.6%    ~29%       ~29%      ~29%
Problem          21.0%    ~21%       ~21%      ~21%
Change           10.2%    ~10%       ~10%      ~10%
```

### Tests

Verify:

```text
every class appears in every split
```

and:

```text
proportions stay close
```

### Exit criteria

Class distribution is preserved across all three splits.

---

## Step 2.2.6 — Add Integrity Checks

Before saving anything, verify the split is valid.

This is one of the most important engineering steps.

### 1. No overlap

```text
train ∩ validation = empty

train ∩ test = empty

validation ∩ test = empty
```

### 2. No missing records

```text
train + validation + test
=
all usable Phase 2.1 records
```

For the current dataset:

```text
total expected = 28,587
```

### 3. No duplicate assignment

Every `record_id` must belong to exactly one split.

### 4. Correct dataset version

Must remain:

```text
ds_6fa2f19cf683
```

### 5. Target distribution present

Every class should have expected counts.

### Tests

This step should test:

- no overlap
- deterministic split
- class preservation where expected

### Exit criteria

The split is mathematically and structurally valid.

---

## Step 2.2.7 — Persist and Freeze the Split

Only after the split passes checks should you save it.

Suggested structure:

```text
artifacts/phase2/splits/

train.jsonl
validation.jsonl
test.jsonl
split_manifest.json
```

The manifest should include at least:

```text
dataset_version
random_seed
split policy
split ratios

train_count
validation_count
test_count

class counts per split

record IDs
```

### Final test

Run the split twice.

You should get:

```text
same train.jsonl
same validation.jsonl
same test.jsonl
same split_manifest.json
```

or at least exactly the same record assignments.

### Exit criteria

Phase 2.2 is complete when:

```text
✓ no train/test leakage
✓ reproducible
✓ class distributions documented
✓ same split reusable by every model
```

---

# Recommended Learning Sequence

Follow it exactly like this:

```text
2.2.1
Understand Train / Validation / Test
        ↓
2.2.2
Understand Stratification
        ↓
2.2.3
Create Split Configuration
        ↓
2.2.4
Build Deterministic Split
        ↓
2.2.5
Add Stratification
        ↓
2.2.6
Add Integrity Checks
        ↓
2.2.7
Persist + Freeze Split
```

This keeps each step focused on **one new idea**.

---

# Roadmap Position

```text
Phase 2.1 Dataset Audit       ✅

Phase 2.2 Dataset Split       ← NOW
    2.2.1 Understand splits
    2.2.2 Understand stratification
    2.2.3 Split configuration
    2.2.4 Deterministic split
    2.2.5 Stratification
    2.2.6 Integrity validation
    2.2.7 Persist/freeze artifacts
           ↓

Phase 2.3 TF-IDF
           ↓
Phase 2.4 Logistic Regression
```

After Phase 2.2, do **not** train a model yet. The next step is **Phase 2.3 — TF-IDF Feature Pipeline**.

Recommended starting point: **2.2.1 only — understand Train vs Validation vs Test before writing code.**

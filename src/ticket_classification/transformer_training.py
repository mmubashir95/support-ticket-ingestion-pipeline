"""Explicit, reproducible Phase 2.8 DistilBERT fine-tuning workflow."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from ticket_classification.evaluation import evaluate_predictions
from ticket_classification.transformer import DEFAULT_LABEL_ORDER, DEFAULT_TRANSFORMER_MODEL, TransformerRecord
from ticket_classification.transformer_outputs import (
    CONFIG_FILENAME,
    LABEL_MAPPING_FILENAME,
    MANIFEST_FILENAME,
    TEST_FILENAME,
    TOKENIZER_POLICY_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    LabelMapping,
    TokenizerPolicy,
    TransformerDatasetConfig,
    TransformerDatasetManifest,
    load_transformer_manifest,
    load_transformer_records,
)
from ticket_pipeline.versioning import fingerprint_value


EXPECTED_DATASET_VERSION = "ds_6fa2f19cf683"
EXPECTED_SPLIT_VERSION = "split_d1f67d41500f"
EXPECTED_TRANSFORMER_DATASET_VERSION = "transformer_ds_340aff24d262"
EXPECTED_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
EXPECTED_COUNTS = {"train": 20_010, "validation": 4_288, "test": 4_289}
EXPECTED_RECORD_ID_FINGERPRINTS = {
    "train": "b734c1fa7aa48006ae66930ef77b95746e97f2dea7c6a8aa312c34546e589668",
    "validation": "c46f77b0953da14e5f860861aa3f02e463e4b0e8a63eaed029110ed2599187fe",
    "test": "7d851ba86275db4bec5bc56e76fe116b6076c1c63a51d0b886af483d98e98597",
}
EXPECTED_TOKENIZED_FINGERPRINTS = {
    "train": "d5ad94ff5c6f2487a7c7c6598255f83ab5a0041827e59c716d526afee8df1d91",
    "validation": "4a2dbad96acf4de726f185bb460f742931c1f7dbc4515abf1f925e016a050623",
    "test": "550748c22fca8680d869e1d069e69e02088a30d637485ab1fb8a8786cb08ef57",
}


class TransformerTrainingConfig(BaseModel):
    """Frozen unweighted baseline configuration for the user-run training job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: Literal[DEFAULT_TRANSFORMER_MODEL] = DEFAULT_TRANSFORMER_MODEL
    model_revision: Literal[EXPECTED_MODEL_REVISION] = EXPECTED_MODEL_REVISION
    tokenizer_name: Literal[DEFAULT_TRANSFORMER_MODEL] = DEFAULT_TRANSFORMER_MODEL
    tokenizer_revision: Literal[EXPECTED_MODEL_REVISION] = EXPECTED_MODEL_REVISION
    dataset_version: Literal[EXPECTED_DATASET_VERSION] = EXPECTED_DATASET_VERSION
    split_version: Literal[EXPECTED_SPLIT_VERSION] = EXPECTED_SPLIT_VERSION
    transformer_dataset_version: Literal[EXPECTED_TRANSFORMER_DATASET_VERSION] = EXPECTED_TRANSFORMER_DATASET_VERSION
    num_labels: Literal[4] = 4
    label_order: tuple[StrictStr, ...] = DEFAULT_LABEL_ORDER
    epochs: StrictInt = Field(default=3, gt=0)
    train_batch_size: StrictInt = Field(default=16, gt=0)
    eval_batch_size: StrictInt = Field(default=32, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    scheduler: Literal["linear"] = "linear"
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    random_seed: StrictInt = Field(default=42, ge=0, le=4294967295)
    max_length: Literal[256] = 256
    padding: Literal["dynamic_at_batch_time"] = "dynamic_at_batch_time"
    evaluation_strategy: Literal["epoch"] = "epoch"
    logging_steps: StrictInt = Field(default=50, gt=0)
    checkpoint_strategy: Literal["epoch"] = "epoch"
    best_model_metric: Literal["macro_f1"] = "macro_f1"
    greater_is_better: Literal[True] = True
    early_stopping_enabled: StrictBool = False
    early_stopping_patience: StrictInt = Field(default=2, gt=0)
    class_weight: None = None
    device_preference: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    gradient_clip_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_contract(self) -> "TransformerTrainingConfig":
        if self.label_order != DEFAULT_LABEL_ORDER:
            raise ValueError("label_order must preserve the Phase 2.7 order")
        return self


class TrainingHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: float
    step: StrictInt
    train_loss: float | None
    validation_loss: float | None
    learning_rate: float
    validation_accuracy: float | None
    validation_macro_precision: float | None
    validation_macro_recall: float | None
    validation_macro_f1: float | None
    validation_weighted_f1: float | None


class BestCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: StrictStr
    epoch: float
    step: StrictInt
    validation_macro_f1: float


class TokenizedDataset:
    """Thin immutable adapter over Phase 2.7 records."""

    def __init__(self, records: Sequence[TransformerRecord], split_name: str):
        self._records = tuple(records)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        return {
            "record_id": record.record_id,
            "input_ids": list(record.input_ids),
            "attention_mask": list(record.attention_mask),
            "labels": record.label,
        }


class DynamicPaddingCollator:
    """Pad each batch only to its longest persisted sequence."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not rows:
            raise ValueError("cannot collate an empty batch")
        maximum = max(len(row["input_ids"]) for row in rows)
        if maximum > 256:
            raise ValueError("batch contains a sequence beyond frozen max_length")
        input_ids = torch.full((len(rows), maximum), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), maximum), dtype=torch.long)
        labels = torch.empty(len(rows), dtype=torch.long)
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            if length != len(row["attention_mask"]):
                raise ValueError("input_ids and attention_mask are misaligned")
            input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = torch.tensor(row["attention_mask"], dtype=torch.long)
            labels[index] = row["labels"]
        return {
            "record_ids": [row["record_id"] for row in rows],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass(frozen=True, slots=True)
class TrainingData:
    train: list[TransformerRecord]
    validation: list[TransformerRecord]
    manifest: TransformerDatasetManifest
    dataset_config: TransformerDatasetConfig
    tokenizer_policy: TokenizerPolicy
    labels: LabelMapping


@dataclass(frozen=True, slots=True)
class ValidationResult:
    loss: float
    metrics: dict[str, Any]
    confusion_matrix: dict[str, Any]
    predictions: list[dict[str, Any]]


def load_training_data(dataset_dir: str | Path, config: TransformerTrainingConfig) -> TrainingData:
    """Load train/validation only; the test JSONL is deliberately untouched."""

    directory = Path(dataset_dir)
    manifest = load_transformer_manifest(directory / MANIFEST_FILENAME)
    dataset_config = TransformerDatasetConfig.model_validate_json((directory / CONFIG_FILENAME).read_text())
    tokenizer_policy = TokenizerPolicy.model_validate_json((directory / TOKENIZER_POLICY_FILENAME).read_text())
    labels = LabelMapping.model_validate_json((directory / LABEL_MAPPING_FILENAME).read_text())
    _validate_phase27_contract(manifest, dataset_config, tokenizer_policy, labels, config)
    train = load_transformer_records(directory / TRAIN_FILENAME)
    validation = load_transformer_records(directory / VALIDATION_FILENAME)
    _validate_group(train, "train", manifest, config)
    _validate_group(validation, "validation", manifest, config)
    if set(record.record_id for record in train) & set(record.record_id for record in validation):
        raise ValueError("train and validation record IDs overlap")
    return TrainingData(train, validation, manifest, dataset_config, tokenizer_policy, labels)


def load_test_data(dataset_dir: str | Path, config: TransformerTrainingConfig) -> tuple[list[TransformerRecord], TrainingData]:
    """Load test only for the separate final evaluation command."""

    training = load_training_data(dataset_dir, config)
    test = load_transformer_records(Path(dataset_dir) / TEST_FILENAME)
    _validate_group(test, "test", training.manifest, config)
    training_ids = {record.record_id for record in training.train + training.validation}
    if training_ids & {record.record_id for record in test}:
        raise ValueError("test record IDs overlap training or validation")
    return test, training


def _validate_phase27_contract(manifest, dataset_config, tokenizer_policy, labels, config) -> None:
    checks = {
        "dataset_version": (manifest.dataset_version, config.dataset_version),
        "split_version": (manifest.split_version, config.split_version),
        "transformer_dataset_version": (manifest.transformer_dataset_version, config.transformer_dataset_version),
        "pretrained_model_name": (manifest.pretrained_model_name, config.model_name),
        "tokenizer_name": (manifest.tokenizer_name, config.tokenizer_name),
        "tokenizer_revision": (tokenizer_policy.revision, config.tokenizer_revision),
        "max_length": (manifest.max_length, config.max_length),
        "label_order": (tuple(manifest.label_order), config.label_order),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Phase 2.7 {name} mismatch: expected={expected!r}, actual={actual!r}")
    if labels.label_order != config.label_order or labels.label_to_id != {
        label: index for index, label in enumerate(config.label_order)
    }:
        raise ValueError("Phase 2.7 label mapping differs from training configuration")
    actual_counts = {
        "train": manifest.train_count,
        "validation": manifest.validation_count,
        "test": manifest.test_count,
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"Phase 2.7 counts differ from frozen contract: {actual_counts}")
    if manifest.record_id_fingerprints != EXPECTED_RECORD_ID_FINGERPRINTS:
        raise ValueError("Phase 2.7 record-ID fingerprints differ from frozen contract")
    if manifest.tokenized_record_fingerprints != EXPECTED_TOKENIZED_FINGERPRINTS:
        raise ValueError("Phase 2.7 tokenized fingerprints differ from frozen contract")
    if dataset_config.padding != "dynamic" or manifest.padding != "dynamic_at_batch_time":
        raise ValueError("Phase 2.7 dynamic-padding policy mismatch")


def _validate_group(records, name, manifest, config) -> None:
    if len(records) != EXPECTED_COUNTS[name]:
        raise ValueError(f"{name} count differs from frozen contract")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{name} contains duplicate record IDs")
    if any(record.label not in range(config.num_labels) for record in records):
        raise ValueError(f"{name} contains an unknown label ID")
    if any(len(record.input_ids) > config.max_length for record in records):
        raise ValueError(f"{name} contains a sequence beyond max_length")
    fingerprint = fingerprint_value([record.model_dump(mode="json") for record in records])
    if fingerprint != manifest.tokenized_record_fingerprints[name]:
        raise ValueError(f"{name} tokenized content fingerprint mismatch")


def resolve_device(preference: str = "auto"):
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if preference != "auto":
        if not available[preference]:
            raise RuntimeError(f"requested device {preference!r} is unavailable")
        return torch.device(preference)
    for name in ("cuda", "mps", "cpu"):
        if available[name]:
            return torch.device(name)
    raise RuntimeError("no supported PyTorch device is available")


def set_reproducible_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(config: TransformerTrainingConfig):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label2id = {label: index for index, label in enumerate(config.label_order)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, revision=config.tokenizer_revision
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=config.num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    validate_model_configuration(model, config)
    return model, tokenizer


def validate_model_configuration(model, config: TransformerTrainingConfig) -> None:
    expected_label2id = {label: index for index, label in enumerate(config.label_order)}
    expected_id2label = {index: label for label, index in expected_label2id.items()}
    if model.config.num_labels != config.num_labels:
        raise ValueError("model num_labels differs from frozen configuration")
    if model.config.label2id != expected_label2id or model.config.id2label != expected_id2label:
        raise ValueError("model label mapping differs from frozen configuration")
    if getattr(model.config, "problem_type", None) not in (None, "single_label_classification"):
        raise ValueError("model problem_type is not single-label classification")


def build_data_loader(records, batch_size, collator, *, shuffle, seed):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        TokenizedDataset(records, "train" if shuffle else "evaluation"),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )


def compute_transformer_metrics(actual_ids, predicted_ids, label_order):
    actual = [label_order[index] for index in actual_ids]
    predicted = [label_order[index] for index in predicted_ids]
    return evaluate_predictions(actual, predicted, list(label_order))


def evaluate_model(model, loader, device, label_order) -> ValidationResult:
    import torch

    model.eval()
    total_loss = 0.0
    total_records = 0
    actual_ids, predicted_ids, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            record_ids = batch.pop("record_ids")
            device_batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**device_batch)
            probabilities = torch.softmax(output.logits, dim=-1)
            predictions = output.logits.argmax(dim=-1)
            count = len(record_ids)
            total_loss += float(output.loss.detach().cpu()) * count
            total_records += count
            labels = device_batch["labels"].detach().cpu().tolist()
            predicted = predictions.detach().cpu().tolist()
            logits = output.logits.detach().cpu().tolist()
            probs = probabilities.detach().cpu().tolist()
            actual_ids.extend(labels)
            predicted_ids.extend(predicted)
            rows.extend(
                {
                    "record_id": record_id,
                    "true_label_id": true_id,
                    "true_label": label_order[true_id],
                    "predicted_label_id": predicted_id,
                    "predicted_label": label_order[predicted_id],
                    "logits": logit,
                    "probabilities": probability,
                }
                for record_id, true_id, predicted_id, logit, probability in zip(
                    record_ids, labels, predicted, logits, probs
                )
            )
    metrics, confusion = compute_transformer_metrics(actual_ids, predicted_ids, label_order)
    return ValidationResult(total_loss / total_records, metrics, confusion, rows)


def select_best_checkpoint(current: BestCheckpoint | None, candidate: BestCheckpoint) -> BestCheckpoint:
    if current is None or candidate.validation_macro_f1 > current.validation_macro_f1:
        return candidate
    return current


def print_training_summary(config, data, device, output_dir, *, smoke_test=False, train_count=None, validation_count=None):
    train_count = train_count or len(data.train)
    validation_count = validation_count or len(data.validation)
    batches = math.ceil(train_count / config.train_batch_size)
    total_steps = batches * config.epochs
    title = "SMOKE TEST MODE" if smoke_test else "Transformer Training Configuration"
    print("=" * 72)
    print(title)
    print("=" * 72)
    values = [
        ("Model", f"{config.model_name}@{config.model_revision}"),
        ("Tokenizer", f"{config.tokenizer_name}@{config.tokenizer_revision}"),
        ("Dataset", config.dataset_version),
        ("Split", config.split_version),
        ("Transformer dataset", config.transformer_dataset_version),
        ("Device", str(device).upper()),
        ("Train examples", train_count),
        ("Validation examples", validation_count),
        ("Epochs", config.epochs),
        ("Train batch size", config.train_batch_size),
        ("Evaluation batch size", config.eval_batch_size),
        ("Batches per epoch", batches),
        ("Total optimization steps", total_steps),
        ("Learning rate", config.learning_rate),
        ("Weight decay", config.weight_decay),
        ("Warmup", f"{config.warmup_ratio:.1%} of steps"),
        ("Max length", config.max_length),
        ("Padding", config.padding),
        ("Random seed", config.random_seed),
        ("Primary metric", config.best_model_metric),
        ("Checkpoint policy", "every epoch; best by validation macro_f1"),
        ("Early stopping", "disabled" if not config.early_stopping_enabled else f"patience={config.early_stopping_patience}"),
        ("Class weighting", "disabled"),
        ("Output", Path(output_dir).resolve()),
    ]
    for name, value in values:
        print(f"{name}: {value}")
    print("=" * 72, flush=True)


def validate_only(dataset_dir, output_dir, config=None) -> dict[str, Any]:
    """Validate the complete setup with one forward pass and no optimizer."""

    import torch

    settings = config or TransformerTrainingConfig()
    set_reproducible_seed(settings.random_seed)
    data = load_training_data(dataset_dir, settings)
    device = resolve_device(settings.device_preference)
    model, tokenizer = load_model_and_tokenizer(settings)
    collator = DynamicPaddingCollator(tokenizer.pad_token_id)
    loader = build_data_loader(data.train[:2], 2, collator, shuffle=False, seed=settings.random_seed)
    batch = next(iter(loader))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.to(device).eval()
    with torch.no_grad():
        output = model(**{key: value.to(device) for key, value in batch.items() if key != "record_ids"})
    if output.logits.shape != (2, settings.num_labels) or output.loss is None:
        raise RuntimeError("validate-only forward pass returned invalid logits or loss")
    if any(not torch.equal(old, new.detach().cpu()) for old, new in zip(before, model.parameters())):
        raise RuntimeError("validate-only changed model parameters")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    probe = target / ".write-probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    result = {
        "status": "passed",
        "device": str(device),
        "batch_shape": list(batch["input_ids"].shape),
        "logits_shape": list(output.logits.shape),
        "loss_produced": True,
        "optimizer_steps": 0,
        "parameters_unchanged": True,
        "test_set_loaded": False,
    }
    print_training_summary(settings, data, device, output_dir)
    print("VALIDATE-ONLY PASSED", json.dumps(result, sort_keys=True), flush=True)
    return result


def run_training(
    dataset_dir,
    output_dir,
    config=None,
    *,
    smoke_test=False,
    model_and_tokenizer=None,
) -> dict[str, Any]:
    """Train on train, select on validation, and never load test records."""

    import torch
    import transformers

    settings = config or TransformerTrainingConfig()
    if smoke_test:
        settings = settings.model_copy(
            update={"epochs": 1, "train_batch_size": 8, "eval_batch_size": 8, "logging_steps": 1}
        )
    set_reproducible_seed(settings.random_seed)
    data = load_training_data(dataset_dir, settings)
    train_records = data.train[:32] if smoke_test else data.train
    validation_records = data.validation[:16] if smoke_test else data.validation
    device = resolve_device(settings.device_preference)
    model, tokenizer = model_and_tokenizer or load_model_and_tokenizer(settings)
    validate_model_configuration(model, settings)
    model.to(device)
    collator = DynamicPaddingCollator(tokenizer.pad_token_id)
    train_loader = build_data_loader(
        train_records, settings.train_batch_size, collator, shuffle=True, seed=settings.random_seed
    )
    validation_loader = build_data_loader(
        validation_records, settings.eval_batch_size, collator, shuffle=False, seed=settings.random_seed
    )
    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    best_model_dir = output / "best_model"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    print_training_summary(
        settings, data, device, output, smoke_test=smoke_test,
        train_count=len(train_records), validation_count=len(validation_records),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    total_steps = len(train_loader) * settings.epochs
    warmup_steps = int(total_steps * settings.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    history: list[TrainingHistoryEntry] = []
    best: BestCheckpoint | None = None
    no_improvement = 0
    global_step = 0
    start = perf_counter()
    for epoch_index in range(settings.epochs):
        model.train()
        running_loss = 0.0
        epoch_records = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch.pop("record_ids")
            device_batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**device_batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            count = len(device_batch["labels"])
            running_loss += float(result.loss.detach().cpu()) * count
            epoch_records += count
            if global_step % settings.logging_steps == 0 or batch_index == len(train_loader):
                elapsed = max(perf_counter() - start, 1e-12)
                print(
                    f"epoch={epoch_index + 1}/{settings.epochs} step={global_step}/{total_steps} "
                    f"train_loss={running_loss / epoch_records:.6f} "
                    f"learning_rate={scheduler.get_last_lr()[0]:.8g} "
                    f"steps_per_second={global_step / elapsed:.3f}",
                    flush=True,
                )
        evaluation = evaluate_model(model, validation_loader, device, settings.label_order)
        metrics = evaluation.metrics
        checkpoint_dir = checkpoints / f"checkpoint-{global_step}"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_index + 1, "step": global_step},
            checkpoint_dir / "training_state.pt",
        )
        candidate = BestCheckpoint(
            path=str(checkpoint_dir.resolve()),
            epoch=float(epoch_index + 1),
            step=global_step,
            validation_macro_f1=metrics["macro_f1"],
        )
        selected = select_best_checkpoint(best, candidate)
        improved = selected is candidate
        if improved:
            best = candidate
            no_improvement = 0
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            model.save_pretrained(best_model_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_model_dir)
        else:
            no_improvement += 1
        history.append(
            TrainingHistoryEntry(
                epoch=float(epoch_index + 1), step=global_step,
                train_loss=running_loss / epoch_records,
                validation_loss=evaluation.loss,
                learning_rate=scheduler.get_last_lr()[0],
                validation_accuracy=metrics["accuracy"],
                validation_macro_precision=metrics["macro_precision"],
                validation_macro_recall=metrics["macro_recall"],
                validation_macro_f1=metrics["macro_f1"],
                validation_weighted_f1=metrics["weighted_f1"],
            )
        )
        print(
            f"validation epoch={epoch_index + 1} loss={evaluation.loss:.6f} "
            f"accuracy={metrics['accuracy']:.6f} macro_precision={metrics['macro_precision']:.6f} "
            f"macro_recall={metrics['macro_recall']:.6f} macro_f1={metrics['macro_f1']:.6f} "
            f"weighted_f1={metrics['weighted_f1']:.6f} best_macro_f1={best.validation_macro_f1:.6f} "
            f"best_checkpoint={best.path}", flush=True,
        )
        _write_json(output / "training_history.json", [item.model_dump(mode="json") for item in history])
        _write_json(output / "best_checkpoint.json", best.model_dump(mode="json"))
        if settings.early_stopping_enabled and no_improvement >= settings.early_stopping_patience:
            print(f"early_stopping step={global_step} patience={settings.early_stopping_patience}", flush=True)
            break
    if best is None:
        raise RuntimeError("training completed without a checkpoint")
    from transformers import AutoModelForSequenceClassification

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    validate_model_configuration(best_model, settings)
    final_validation = evaluate_model(best_model, validation_loader, device, settings.label_order)
    _write_json(output / "training_config.json", settings.model_dump(mode="json"))
    _write_json(output / "validation_metrics.json", {"loss": final_validation.loss, **final_validation.metrics})
    _write_json(output / "validation_confusion_matrix.json", final_validation.confusion_matrix)
    _write_jsonl(output / "validation_predictions.jsonl", final_validation.predictions)
    identity = {
        "dataset_version": settings.dataset_version,
        "split_version": settings.split_version,
        "transformer_dataset_version": settings.transformer_dataset_version,
        "configuration": settings.model_dump(mode="json"),
        "smoke_test": smoke_test,
    }
    manifest = {
        **identity,
        "training_run_id": ("smoke_" if smoke_test else "distilbert_") + fingerprint_value(identity)[:12],
        "best_checkpoint": best.model_dump(mode="json"),
        "resolved_device": str(device),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "test_set_loaded": False,
        "training_seconds": perf_counter() - start,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "platform": platform.platform(),
        },
        "reproducibility": "Python, NumPy and PyTorch seeds are set; exact GPU/MPS bitwise determinism is not claimed.",
        "class_weighting": "disabled",
        "loss": "unweighted cross_entropy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output.resolve()),
    }
    _write_json(output / "training_manifest.json", manifest)
    return manifest


def run_final_test_evaluation(dataset_dir, training_dir, config=None) -> dict[str, Any]:
    """Evaluate the already selected best model on test without retraining."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = config or TransformerTrainingConfig()
    training_root = Path(training_dir)
    best_model_dir = training_root / "best_model"
    if not best_model_dir.is_dir():
        raise FileNotFoundError(f"best model directory does not exist: {best_model_dir}")
    training_manifest = json.loads((training_root / "training_manifest.json").read_text())
    if training_manifest.get("smoke_test"):
        raise ValueError("smoke-test checkpoint cannot be used for final test evaluation")
    if training_manifest.get("test_set_loaded") is not False:
        raise ValueError("training manifest does not prove test isolation")
    expected_identity = {
        "dataset_version": settings.dataset_version,
        "split_version": settings.split_version,
        "transformer_dataset_version": settings.transformer_dataset_version,
        "train_count": EXPECTED_COUNTS["train"],
        "validation_count": EXPECTED_COUNTS["validation"],
    }
    for name, expected in expected_identity.items():
        if training_manifest.get(name) != expected:
            raise ValueError(
                f"training manifest {name} mismatch: "
                f"expected={expected!r}, actual={training_manifest.get(name)!r}"
            )
    if training_manifest.get("configuration") != settings.model_dump(mode="json"):
        raise ValueError("training manifest configuration differs from final evaluation")
    test, _ = load_test_data(dataset_dir, settings)
    device = resolve_device(settings.device_preference)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    validate_model_configuration(model, settings)
    loader = build_data_loader(
        test, settings.eval_batch_size, DynamicPaddingCollator(tokenizer.pad_token_id),
        shuffle=False, seed=settings.random_seed,
    )
    result = evaluate_model(model, loader, device, settings.label_order)
    _write_json(training_root / "test_metrics.json", {"loss": result.loss, **result.metrics})
    _write_json(training_root / "test_confusion_matrix.json", result.confusion_matrix)
    _write_jsonl(training_root / "test_predictions.jsonl", result.predictions)
    report = {
        "checkpoint": str(best_model_dir.resolve()),
        "dataset_version": settings.dataset_version,
        "split_version": settings.split_version,
        "transformer_dataset_version": settings.transformer_dataset_version,
        "test_count": len(test),
        "metrics": {"loss": result.loss, **result.metrics},
        "inference_only": True,
        "checkpoint_selection_changed": False,
        "device": str(device),
    }
    _write_json(training_root / "test_evaluation_manifest.json", report)
    print("FINAL TEST EVALUATION", json.dumps(report["metrics"], sort_keys=True), flush=True)
    return report


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

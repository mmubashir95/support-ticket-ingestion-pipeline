"""Transformer tokenization for the existing frozen classification splits."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from ticket_classification.features import (
    CLASSIFICATION_INPUT_FIELDS,
    CLASSIFICATION_INPUT_POLICY,
    CLASSIFICATION_TEXT_SEPARATOR,
    FrozenSplits,
    build_classification_text,
)
from ticket_classification.models import ClassificationRecord


DEFAULT_TRANSFORMER_MODEL = "distilbert/distilbert-base-uncased"
DEFAULT_LABEL_ORDER = ("Change", "Incident", "Problem", "Request")
TOKEN_LENGTH_CANDIDATES = (64, 128, 256, 512)


class TokenizerLike(Protocol):
    """Small tokenizer surface used by preparation and local unit fixtures."""

    name_or_path: str
    model_max_length: int
    pad_token: str | None
    pad_token_id: int | None
    special_tokens_map: dict[str, Any]
    init_kwargs: dict[str, Any]

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]: ...


class TransformerDatasetConfig(BaseModel):
    """Validated identity and tokenization policy for Phase 2.7."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pretrained_model_name: Literal[DEFAULT_TRANSFORMER_MODEL] = DEFAULT_TRANSFORMER_MODEL
    tokenizer_name: Literal[DEFAULT_TRANSFORMER_MODEL] = DEFAULT_TRANSFORMER_MODEL
    dataset_version: StrictStr
    input_fields: tuple[Literal["subject", "message"], Literal["subject", "message"]] = CLASSIFICATION_INPUT_FIELDS
    input_policy: Literal["subject + message"] = CLASSIFICATION_INPUT_POLICY
    text_separator: Literal["\n\n"] = CLASSIFICATION_TEXT_SEPARATOR
    label_order: tuple[StrictStr, ...] = DEFAULT_LABEL_ORDER
    max_length: StrictInt | None = Field(default=None, ge=1, le=512)
    truncation: Literal[True] = True
    padding: Literal["dynamic"] = "dynamic"
    add_special_tokens: Literal[True] = True

    @model_validator(mode="after")
    def validate_contract(self) -> "TransformerDatasetConfig":
        if not self.dataset_version.strip():
            raise ValueError("dataset_version must be non-blank")
        if tuple(self.input_fields) != CLASSIFICATION_INPUT_FIELDS:
            raise ValueError("input_fields must preserve the frozen subject + message policy")
        if not self.label_order or len(set(self.label_order)) != len(self.label_order):
            raise ValueError("label_order must contain unique labels")
        return self


class LabelMapping(BaseModel):
    """Deterministic reversible mapping for the frozen class order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_order: tuple[StrictStr, ...]
    label_to_id: dict[StrictStr, int]
    id_to_label: dict[int, StrictStr]

    @model_validator(mode="after")
    def validate_bijection(self) -> "LabelMapping":
        expected = {label: index for index, label in enumerate(self.label_order)}
        reverse = {index: label for label, index in expected.items()}
        if self.label_to_id != expected or self.id_to_label != reverse:
            raise ValueError("label mappings must exactly follow label_order")
        return self

    def encode(self, label: str) -> int:
        try:
            return self.label_to_id[label]
        except KeyError as error:
            raise ValueError(f"unknown classification label: {label!r}") from error

    def decode(self, label_id: int) -> str:
        try:
            return self.id_to_label[label_id]
        except KeyError as error:
            raise ValueError(f"unknown classification label id: {label_id!r}") from error


class TokenLengthCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_length: StrictInt
    covered_count: StrictInt
    covered_percentage: float
    truncated_count: StrictInt
    truncated_percentage: float


class TokenLengthAnalysis(BaseModel):
    """Observed unpadded, untruncated lengths including special tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: StrictInt
    minimum: StrictInt
    mean: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    maximum: StrictInt
    includes_special_tokens: Literal[True] = True
    padding_applied: Literal[False] = False
    truncation_applied: Literal[False] = False
    coverage: dict[StrictStr, TokenLengthCoverage]


class MaxLengthDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_max_length: StrictInt
    retained_count: StrictInt
    retained_percentage: float
    truncated_count: StrictInt
    truncated_percentage: float
    rationale: StrictStr


class TransformerRecord(BaseModel):
    """Variable-length tokenized record ready for later dynamic padding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: StrictStr
    input_ids: list[StrictInt] = Field(min_length=1)
    attention_mask: list[StrictInt] = Field(min_length=1)
    label: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_alignment(self) -> "TransformerRecord":
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("attention_mask length must equal input_ids length")
        if any(value not in (0, 1) for value in self.attention_mask):
            raise ValueError("attention_mask must contain only zero or one")
        return self


@dataclass(frozen=True, slots=True)
class TransformerDatasets:
    train: list[TransformerRecord]
    validation: list[TransformerRecord]
    test: list[TransformerRecord]


def build_label_mapping(label_order: Sequence[str]) -> LabelMapping:
    order = tuple(label_order)
    return LabelMapping(
        label_order=order,
        label_to_id={label: index for index, label in enumerate(order)},
        id_to_label={index: label for index, label in enumerate(order)},
    )


def load_tokenizer(config: TransformerDatasetConfig) -> TokenizerLike:
    """Load the matching Hugging Face tokenizer without loading a model."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for Phase 2.7 tokenization") from error
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    validate_tokenizer(tokenizer, config)
    return tokenizer


def validate_tokenizer(tokenizer: TokenizerLike, config: TransformerDatasetConfig) -> None:
    if str(tokenizer.name_or_path) != config.tokenizer_name:
        raise ValueError(
            "loaded tokenizer identity does not match configured tokenizer_name"
        )
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a padding token for dynamic batching")
    if not tokenizer.special_tokens_map:
        raise ValueError("tokenizer must expose its special-token behavior")
    encoded = tokenizer("tokenizer validation", add_special_tokens=True, padding=False, truncation=False)
    if not encoded.get("input_ids") or not encoded.get("attention_mask"):
        raise ValueError("tokenizer must produce input_ids and attention_mask")
    if len(encoded["input_ids"]) != len(encoded["attention_mask"]):
        raise ValueError("tokenizer produced misaligned IDs and attention mask")
    if config.max_length is not None and config.max_length > tokenizer.model_max_length:
        raise ValueError("max_length exceeds tokenizer model_max_length")


def analyze_token_lengths(
    records: Sequence[ClassificationRecord], tokenizer: TokenizerLike
) -> tuple[TokenLengthAnalysis, list[int]]:
    """Measure actual tokenizer lengths without truncation or padding."""

    if not records:
        raise ValueError("records must not be empty")
    lengths: list[int] = []
    for record in records:
        encoded = tokenizer(
            build_classification_text(record),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.append(len(encoded["input_ids"]))
    ordered = sorted(lengths)
    total = len(ordered)
    coverage = {}
    for candidate in TOKEN_LENGTH_CANDIDATES:
        covered = sum(length <= candidate for length in ordered)
        coverage[str(candidate)] = TokenLengthCoverage(
            max_length=candidate,
            covered_count=covered,
            covered_percentage=covered / total,
            truncated_count=total - covered,
            truncated_percentage=(total - covered) / total,
        )
    return TokenLengthAnalysis(
        record_count=total,
        minimum=ordered[0],
        mean=sum(ordered) / total,
        p50=_percentile(ordered, 0.50),
        p75=_percentile(ordered, 0.75),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        coverage=coverage,
    ), lengths


def select_max_length(analysis: TokenLengthAnalysis) -> MaxLengthDecision:
    """Choose the smallest standard length retaining at least 95% of records."""

    selected = 512
    for candidate in (128, 256, 512):
        if analysis.coverage[str(candidate)].covered_percentage >= 0.95:
            selected = candidate
            break
    coverage = analysis.coverage[str(selected)]
    rationale = (
        f"Selected the smallest candidate among 128/256/512 that retains at least 95% "
        f"of records. Observed P90={analysis.p90:g}, P95={analysis.p95:g}, "
        f"P99={analysis.p99:g}; {coverage.covered_percentage:.4%} fit without "
        f"truncation. This limits attention cost while remaining within DistilBERT's "
        "512-token positional limit."
    )
    return MaxLengthDecision(
        selected_max_length=selected,
        retained_count=coverage.covered_count,
        retained_percentage=coverage.covered_percentage,
        truncated_count=coverage.truncated_count,
        truncated_percentage=coverage.truncated_percentage,
        rationale=rationale,
    )


def tokenize_records(
    records: Sequence[ClassificationRecord],
    tokenizer: TokenizerLike,
    config: TransformerDatasetConfig,
    labels: LabelMapping,
) -> list[TransformerRecord]:
    if config.max_length is None:
        raise ValueError("max_length must be selected before dataset tokenization")
    output: list[TransformerRecord] = []
    for record in records:
        if not isinstance(record.label, str):
            raise ValueError("Transformer preparation requires one string label per record")
        encoded = tokenizer(
            build_classification_text(record),
            add_special_tokens=config.add_special_tokens,
            padding=False,
            truncation=config.truncation,
            max_length=config.max_length,
            return_attention_mask=True,
        )
        item = TransformerRecord(
            record_id=record.record_id,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            label=labels.encode(record.label),
        )
        if len(item.input_ids) > config.max_length:
            raise ValueError("tokenizer did not respect max_length")
        output.append(item)
    return output


def build_transformer_datasets(
    splits: FrozenSplits,
    tokenizer: TokenizerLike,
    config: TransformerDatasetConfig,
    labels: LabelMapping,
) -> TransformerDatasets:
    datasets = TransformerDatasets(
        train=tokenize_records(splits.train, tokenizer, config, labels),
        validation=tokenize_records(splits.validation, tokenizer, config, labels),
        test=tokenize_records(splits.test, tokenizer, config, labels),
    )
    validate_dataset_alignment(splits, datasets, config, labels)
    return datasets


def validate_dataset_alignment(
    splits: FrozenSplits,
    datasets: TransformerDatasets,
    config: TransformerDatasetConfig,
    labels: LabelMapping,
) -> None:
    if config.dataset_version != splits.manifest.dataset_version:
        raise ValueError("Transformer dataset version does not match frozen split")
    frozen_order = tuple(sorted(splits.manifest.class_counts_full))
    if config.label_order != frozen_order or labels.label_order != frozen_order:
        raise ValueError("Transformer label order does not match frozen class order")
    groups = {"train": datasets.train, "validation": datasets.validation, "test": datasets.test}
    source = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    assigned: list[str] = []
    for name, records in groups.items():
        expected_ids = [record.record_id for record in source[name]]
        actual_ids = [record.record_id for record in records]
        if actual_ids != expected_ids:
            raise ValueError(f"Transformer {name} membership/order differs from frozen split")
        assigned.extend(actual_ids)
        expected_labels = [labels.encode(str(record.label)) for record in source[name]]
        if [record.label for record in records] != expected_labels:
            raise ValueError(f"Transformer {name} labels differ from frozen split")
    if len(assigned) != len(set(assigned)):
        raise ValueError("Transformer datasets contain duplicate record assignment")
    observed = Counter(record.label for records in groups.values() for record in records)
    if set(observed) != set(range(len(labels.label_order))):
        raise ValueError("Transformer datasets do not represent every frozen label")


def _percentile(ordered: Sequence[int], quantile: float) -> float:
    """Linear percentile compatible with NumPy's default interpolation."""

    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

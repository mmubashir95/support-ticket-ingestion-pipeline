"""Phase 2.7 Transformer dataset preparation tests without network access."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ticket_classification.features import FrozenSplits, build_classification_text
from ticket_classification.models import ClassificationRecord
from ticket_classification.split_outputs import write_split_artifacts
from ticket_classification.splitting import SplitConfig, split_classification_records
from ticket_classification.transformer import (
    DEFAULT_LABEL_ORDER,
    DEFAULT_TRANSFORMER_MODEL,
    LabelMapping,
    TransformerDatasetConfig,
    TransformerRecord,
    analyze_token_lengths,
    build_label_mapping,
    build_transformer_datasets,
    load_tokenizer,
    select_max_length,
    tokenize_records,
    validate_dataset_alignment,
    validate_tokenizer,
)
from ticket_classification.transformer_outputs import (
    load_transformer_manifest,
    load_transformer_records,
)
from ticket_classification.workflow import run_transformer_dataset_preparation


DATASET_VERSION = "ds_6fa2f19cf683"


class FakeTokenizer:
    """Deterministic WordPiece-shaped stand-in for unit tests."""

    name_or_path = DEFAULT_TRANSFORMER_MODEL
    model_max_length = 512
    pad_token = "[PAD]"
    pad_token_id = 0
    special_tokens_map = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}
    init_kwargs = {"_commit_hash": "fixture-revision"}

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {key: [self(item, **kwargs)[key] for item in text] for key in ("input_ids", "attention_mask")}
        words = text.split()
        ids = [101] + [1000 + sum(map(ord, word)) % 20000 for word in words] + [102]
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            ids = ids[:max_length]
            if ids:
                ids[-1] = 102
        result = {"input_ids": ids}
        if kwargs.get("return_attention_mask", True):
            result["attention_mask"] = [1] * len(ids)
        return result


def records(per_class: int = 4) -> list[ClassificationRecord]:
    return [
        ClassificationRecord(
            record_id=f"record:{label}:{index}",
            text=f"Subject {label} {index}\n\nMessage " + "detail " * (index + 1),
            label=label,
            source_fields=["subject", "message"],
        )
        for label in DEFAULT_LABEL_ORDER
        for index in range(per_class)
    ]


def frozen_splits() -> FrozenSplits:
    source = records(10)
    config = SplitConfig(dataset_version=DATASET_VERSION)
    result = split_classification_records(source, config)
    from ticket_classification.split_outputs import build_split_manifest
    from ticket_classification.splitting import validate_split_integrity

    report = validate_split_integrity(source, config, result, expected_dataset_version=DATASET_VERSION)
    return FrozenSplits(
        train=result.train,
        validation=result.validation,
        test=result.test,
        manifest=build_split_manifest(result, config, report),
    )


def test_transformer_configuration_freezes_required_policy() -> None:
    config = TransformerDatasetConfig(dataset_version=DATASET_VERSION)
    assert config.pretrained_model_name == DEFAULT_TRANSFORMER_MODEL
    assert config.tokenizer_name == DEFAULT_TRANSFORMER_MODEL
    assert config.input_fields == ("subject", "message")
    assert config.input_policy == "subject + message"
    assert config.label_order == DEFAULT_LABEL_ORDER
    assert config.max_length is None
    assert config.padding == "dynamic"
    assert config.truncation is True
    with pytest.raises(ValidationError):
        TransformerDatasetConfig(dataset_version=DATASET_VERSION, max_length=513)
    with pytest.raises(ValidationError):
        TransformerDatasetConfig(dataset_version=DATASET_VERSION, input_fields=("message", "subject"))


def test_matching_tokenizer_configuration_and_metadata_are_valid() -> None:
    validate_tokenizer(FakeTokenizer(), TransformerDatasetConfig(dataset_version=DATASET_VERSION))
    broken = FakeTokenizer()
    broken.pad_token = None
    with pytest.raises(ValueError, match="padding token"):
        validate_tokenizer(broken, TransformerDatasetConfig(dataset_version=DATASET_VERSION))


def test_loader_requests_exact_matching_hugging_face_tokenizer(monkeypatch) -> None:
    requested = []

    def from_pretrained(name):
        requested.append(name)
        return FakeTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", from_pretrained)
    tokenizer = load_tokenizer(TransformerDatasetConfig(dataset_version=DATASET_VERSION))
    assert tokenizer.name_or_path == DEFAULT_TRANSFORMER_MODEL
    assert requested == [DEFAULT_TRANSFORMER_MODEL]


def test_label_mapping_is_deterministic_reversible_and_rejects_unknowns() -> None:
    mapping = build_label_mapping(DEFAULT_LABEL_ORDER)
    assert mapping.label_to_id == {"Change": 0, "Incident": 1, "Problem": 2, "Request": 3}
    assert mapping.id_to_label == {0: "Change", 1: "Incident", 2: "Problem", 3: "Request"}
    assert [mapping.decode(mapping.encode(label)) for label in DEFAULT_LABEL_ORDER] == list(DEFAULT_LABEL_ORDER)
    with pytest.raises(ValueError, match="unknown classification label"):
        mapping.encode("Other")
    with pytest.raises(ValueError, match="unknown classification label id"):
        mapping.decode(4)
    with pytest.raises(ValidationError):
        LabelMapping(label_order=DEFAULT_LABEL_ORDER, label_to_id={"Change": 1}, id_to_label={1: "Change"})


def test_text_policy_is_reused_without_transformer_preprocessing() -> None:
    record = records(1)[0]
    assert build_classification_text(record) is record.text


def test_token_length_analysis_is_unpadded_untruncated_and_selects_from_evidence() -> None:
    source = records(3)
    analysis, lengths = analyze_token_lengths(source, FakeTokenizer())
    assert analysis.record_count == len(source)
    assert analysis.minimum == min(lengths)
    assert analysis.maximum == max(lengths)
    assert analysis.padding_applied is False
    assert analysis.truncation_applied is False
    assert set(analysis.coverage) == {"64", "128", "256", "512"}
    decision = select_max_length(analysis)
    assert decision.selected_max_length == 128
    assert decision.retained_percentage == 1.0


def test_tokenization_is_deterministic_traceable_and_truncated() -> None:
    source = [ClassificationRecord(record_id="record:long", text="word " * 40, label="Incident", source_fields=["subject", "message"])]
    config = TransformerDatasetConfig(dataset_version=DATASET_VERSION, max_length=16)
    labels = build_label_mapping(DEFAULT_LABEL_ORDER)
    first = tokenize_records(source, FakeTokenizer(), config, labels)
    second = tokenize_records(source, FakeTokenizer(), config, labels)
    assert first == second
    assert first[0].record_id == "record:long"
    assert first[0].label == 1
    assert len(first[0].input_ids) == 16
    assert len(first[0].attention_mask) == len(first[0].input_ids)
    assert all(first[0].attention_mask)


def test_transformer_record_rejects_misaligned_mask() -> None:
    with pytest.raises(ValidationError, match="attention_mask length"):
        TransformerRecord(record_id="record:1", input_ids=[101, 102], attention_mask=[1], label=0)


def test_all_split_membership_counts_labels_and_version_remain_frozen() -> None:
    splits = frozen_splits()
    config = TransformerDatasetConfig(dataset_version=DATASET_VERSION, max_length=128)
    labels = build_label_mapping(DEFAULT_LABEL_ORDER)
    datasets = build_transformer_datasets(splits, FakeTokenizer(), config, labels)
    assert [row.record_id for row in datasets.train] == splits.manifest.train_record_ids
    assert [row.record_id for row in datasets.validation] == splits.manifest.validation_record_ids
    assert [row.record_id for row in datasets.test] == splits.manifest.test_record_ids
    assert (len(datasets.train), len(datasets.validation), len(datasets.test)) == (
        splits.manifest.train_count, splits.manifest.validation_count, splits.manifest.test_count
    )
    assert {row.label for group in (datasets.train, datasets.validation, datasets.test) for row in group} == {0, 1, 2, 3}


def test_alignment_rejects_moved_missing_duplicate_and_wrong_version() -> None:
    splits = frozen_splits()
    labels = build_label_mapping(DEFAULT_LABEL_ORDER)
    config = TransformerDatasetConfig(dataset_version=DATASET_VERSION, max_length=128)
    datasets = build_transformer_datasets(splits, FakeTokenizer(), config, labels)
    datasets.train[0], datasets.validation[0] = datasets.validation[0], datasets.train[0]
    with pytest.raises(ValueError, match="membership/order"):
        validate_dataset_alignment(splits, datasets, config, labels)
    wrong = config.model_copy(update={"dataset_version": "wrong"})
    with pytest.raises(ValueError, match="dataset version"):
        validate_dataset_alignment(splits, build_transformer_datasets(splits, FakeTokenizer(), config, labels), wrong, labels)


def test_workflow_persists_reloadable_byte_reproducible_artifacts(tmp_path: Path) -> None:
    source = records(10)
    split_config = SplitConfig(dataset_version=DATASET_VERSION)
    result = split_classification_records(source, split_config)
    split_dir = tmp_path / "splits"
    write_split_artifacts(source, result, split_config, split_dir, expected_dataset_version=DATASET_VERSION)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_transformer_dataset_preparation(split_dir, first, tokenizer=FakeTokenizer())
    run_transformer_dataset_preparation(split_dir, second, tokenizer=FakeTokenizer())
    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = load_transformer_manifest(first / "transformer_dataset_manifest.json")
    assert manifest.dataset_version == DATASET_VERSION
    assert manifest.label_order == DEFAULT_LABEL_ORDER
    assert manifest.phase_2_8_training_performed is False
    assert manifest.total_count == len(source)
    assert len(load_transformer_records(first / "train_tokenized.jsonl")) == result.train.__len__()
    mapping = json.loads((first / "label_mapping.json").read_text())
    assert mapping["label_to_id"]["Incident"] == 1

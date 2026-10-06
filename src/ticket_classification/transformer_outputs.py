"""Deterministic artifacts for Transformer dataset preparation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from ticket_classification.features import FrozenSplits
from ticket_classification.transformer import (
    LabelMapping,
    MaxLengthDecision,
    TokenLengthAnalysis,
    TokenizerLike,
    TransformerDatasetConfig,
    TransformerDatasets,
    TransformerRecord,
    validate_dataset_alignment,
)
from ticket_pipeline.versioning import fingerprint_value


CONFIG_FILENAME = "transformer_dataset_config.json"
TOKENIZER_POLICY_FILENAME = "tokenizer_policy.json"
LABEL_MAPPING_FILENAME = "label_mapping.json"
TOKEN_ANALYSIS_FILENAME = "token_length_analysis.json"
TOKEN_ANALYSIS_MARKDOWN_FILENAME = "token_length_analysis.md"
MANIFEST_FILENAME = "transformer_dataset_manifest.json"
TRAIN_FILENAME = "train_tokenized.jsonl"
VALIDATION_FILENAME = "validation_tokenized.jsonl"
TEST_FILENAME = "test_tokenized.jsonl"


class TransformerArtifactPersistenceError(RuntimeError):
    """Transformer preparation artifacts failed validation or persistence."""


class TokenizerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer_name: StrictStr
    resolved_name_or_path: StrictStr
    revision: StrictStr | None
    model_max_length: StrictInt
    pad_token: StrictStr
    pad_token_id: StrictInt
    special_tokens_map: dict[StrictStr, Any]
    add_special_tokens: Literal[True] = True
    truncation: Literal[True] = True
    padding: Literal["dynamic_at_batch_time"] = "dynamic_at_batch_time"
    persisted_examples_are_variable_length: Literal[True] = True


class TransformerDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: Literal["2.7"] = "2.7"
    dataset_version: StrictStr
    split_version: StrictStr
    transformer_dataset_version: StrictStr
    pretrained_model_name: StrictStr
    tokenizer_name: StrictStr
    tokenizer_revision: StrictStr | None
    input_policy: StrictStr
    input_fields: tuple[StrictStr, ...]
    label_order: tuple[StrictStr, ...]
    label_to_id: dict[StrictStr, int]
    max_length: StrictInt
    truncation: Literal[True] = True
    padding: Literal["dynamic_at_batch_time"] = "dynamic_at_batch_time"
    train_count: StrictInt
    validation_count: StrictInt
    test_count: StrictInt
    total_count: StrictInt
    class_counts: dict[StrictStr, int]
    record_id_fingerprints: dict[StrictStr, StrictStr]
    tokenized_record_fingerprints: dict[StrictStr, StrictStr]
    token_length_analysis: TokenLengthAnalysis
    max_length_decision: MaxLengthDecision
    split_membership_verified: Literal[True] = True
    phase_2_8_training_performed: Literal[False] = False


@dataclass(frozen=True, slots=True)
class TransformerArtifacts:
    config_path: Path
    tokenizer_policy_path: Path
    label_mapping_path: Path
    token_analysis_path: Path
    token_analysis_markdown_path: Path
    manifest_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path


def build_tokenizer_policy(
    tokenizer: TokenizerLike, config: TransformerDatasetConfig
) -> TokenizerPolicy:
    revision = _resolve_tokenizer_revision(tokenizer, config.tokenizer_name)
    return TokenizerPolicy(
        tokenizer_name=config.tokenizer_name,
        resolved_name_or_path=str(tokenizer.name_or_path),
        revision=str(revision) if revision is not None else None,
        model_max_length=int(tokenizer.model_max_length),
        pad_token=str(tokenizer.pad_token),
        pad_token_id=int(tokenizer.pad_token_id),
        special_tokens_map=dict(tokenizer.special_tokens_map),
    )


def _resolve_tokenizer_revision(
    tokenizer: TokenizerLike, tokenizer_name: str
) -> str | None:
    revision = tokenizer.init_kwargs.get("_commit_hash") or tokenizer.init_kwargs.get("revision")
    if revision is not None:
        return str(revision)
    try:
        from huggingface_hub import scan_cache_dir

        matching = [repo for repo in scan_cache_dir().repos if repo.repo_id == tokenizer_name]
        main_revisions = [
            item.commit_hash
            for repo in matching
            for item in repo.revisions
            if "main" in item.refs
        ]
        if len(main_revisions) == 1:
            return main_revisions[0]
    except (ImportError, OSError):
        pass
    return None


def build_transformer_manifest(
    splits: FrozenSplits,
    datasets: TransformerDatasets,
    config: TransformerDatasetConfig,
    labels: LabelMapping,
    tokenizer_policy: TokenizerPolicy,
    analysis: TokenLengthAnalysis,
    decision: MaxLengthDecision,
) -> TransformerDatasetManifest:
    if config.max_length is None:
        raise ValueError("final Transformer configuration must freeze max_length")
    validate_dataset_alignment(splits, datasets, config, labels)
    split_dump = splits.manifest.model_dump(mode="json")
    split_version = f"split_{fingerprint_value(split_dump)[:12]}"
    groups = {
        "train": datasets.train,
        "validation": datasets.validation,
        "test": datasets.test,
    }
    record_ids = {
        name: [record.record_id for record in records] for name, records in groups.items()
    }
    tokenized = {
        name: [record.model_dump(mode="json") for record in records]
        for name, records in groups.items()
    }
    identity = {
        "dataset_version": config.dataset_version,
        "split_version": split_version,
        "config": config.model_dump(mode="json"),
        "tokenizer_policy": tokenizer_policy.model_dump(mode="json"),
        "record_id_fingerprints": {
            name: fingerprint_value(ids) for name, ids in record_ids.items()
        },
    }
    return TransformerDatasetManifest(
        dataset_version=config.dataset_version,
        split_version=split_version,
        transformer_dataset_version=f"transformer_ds_{fingerprint_value(identity)[:12]}",
        pretrained_model_name=config.pretrained_model_name,
        tokenizer_name=config.tokenizer_name,
        tokenizer_revision=tokenizer_policy.revision,
        input_policy=config.input_policy,
        input_fields=config.input_fields,
        label_order=labels.label_order,
        label_to_id=labels.label_to_id,
        max_length=config.max_length,
        train_count=len(datasets.train),
        validation_count=len(datasets.validation),
        test_count=len(datasets.test),
        total_count=sum(len(records) for records in groups.values()),
        class_counts=splits.manifest.class_counts_full,
        record_id_fingerprints={
            name: fingerprint_value(ids) for name, ids in record_ids.items()
        },
        tokenized_record_fingerprints={
            name: fingerprint_value(rows) for name, rows in tokenized.items()
        },
        token_length_analysis=analysis,
        max_length_decision=decision,
    )


def write_transformer_artifacts(
    splits: FrozenSplits,
    datasets: TransformerDatasets,
    config: TransformerDatasetConfig,
    labels: LabelMapping,
    tokenizer: TokenizerLike,
    analysis: TokenLengthAnalysis,
    decision: MaxLengthDecision,
    output_dir: str | Path,
) -> TransformerArtifacts:
    """Persist tokenized splits and deterministic metadata, then verify reload."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = TransformerArtifacts(
        config_path=directory / CONFIG_FILENAME,
        tokenizer_policy_path=directory / TOKENIZER_POLICY_FILENAME,
        label_mapping_path=directory / LABEL_MAPPING_FILENAME,
        token_analysis_path=directory / TOKEN_ANALYSIS_FILENAME,
        token_analysis_markdown_path=directory / TOKEN_ANALYSIS_MARKDOWN_FILENAME,
        manifest_path=directory / MANIFEST_FILENAME,
        train_path=directory / TRAIN_FILENAME,
        validation_path=directory / VALIDATION_FILENAME,
        test_path=directory / TEST_FILENAME,
    )
    policy = build_tokenizer_policy(tokenizer, config)
    manifest = build_transformer_manifest(
        splits, datasets, config, labels, policy, analysis, decision
    )
    try:
        _atomic_json(artifacts.config_path, config.model_dump(mode="json"))
        _atomic_json(artifacts.tokenizer_policy_path, policy.model_dump(mode="json"))
        _atomic_json(artifacts.label_mapping_path, labels.model_dump(mode="json"))
        _atomic_json(artifacts.token_analysis_path, analysis.model_dump(mode="json"))
        _atomic_text(artifacts.token_analysis_markdown_path, render_token_analysis(analysis, decision))
        _atomic_json(artifacts.manifest_path, manifest.model_dump(mode="json"))
        _atomic_jsonl(artifacts.train_path, datasets.train)
        _atomic_jsonl(artifacts.validation_path, datasets.validation)
        _atomic_jsonl(artifacts.test_path, datasets.test)
        _verify_artifacts(artifacts, splits, datasets, config, labels, policy, analysis, manifest)
    except TransformerArtifactPersistenceError:
        raise
    except Exception as error:
        raise TransformerArtifactPersistenceError(
            "Failed to persist and verify Transformer dataset artifacts."
        ) from error
    return artifacts


def load_transformer_records(path: str | Path) -> list[TransformerRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(TransformerRecord.model_validate_json(line))
            except Exception as error:
                raise TransformerArtifactPersistenceError(
                    f"invalid tokenized record at line {line_number}: {path}"
                ) from error
    return records


def load_transformer_manifest(path: str | Path) -> TransformerDatasetManifest:
    return TransformerDatasetManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def render_token_analysis(analysis: TokenLengthAnalysis, decision: MaxLengthDecision) -> str:
    rows = [
        "# DistilBERT Token-Length Analysis",
        "",
        "Lengths use the matching tokenizer with special tokens, without padding or truncation.",
        "",
        "| Statistic | Tokens |",
        "|---|---:|",
        f"| Minimum | {analysis.minimum} |",
        f"| Mean | {analysis.mean:.4f} |",
        f"| P50 | {analysis.p50:g} |",
        f"| P75 | {analysis.p75:g} |",
        f"| P90 | {analysis.p90:g} |",
        f"| P95 | {analysis.p95:g} |",
        f"| P99 | {analysis.p99:g} |",
        f"| Maximum | {analysis.maximum} |",
        "",
        "| Candidate | Covered | Coverage | Requires truncation | Truncation rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for candidate in (64, 128, 256, 512):
        item = analysis.coverage[str(candidate)]
        rows.append(
            f"| {candidate} | {item.covered_count} | {item.covered_percentage:.4%} | "
            f"{item.truncated_count} | {item.truncated_percentage:.4%} |"
        )
    rows.extend(
        [
            "",
            f"Selected `max_length = {decision.selected_max_length}`.",
            "",
            decision.rationale,
            "",
            "Persisted records remain variable length. Phase 2.8 should apply dynamic padding per batch.",
            "",
        ]
    )
    return "\n".join(rows)


def _verify_artifacts(
    artifacts: TransformerArtifacts,
    splits: FrozenSplits,
    datasets: TransformerDatasets,
    config: TransformerDatasetConfig,
    labels: LabelMapping,
    policy: TokenizerPolicy,
    analysis: TokenLengthAnalysis,
    manifest: TransformerDatasetManifest,
) -> None:
    loaded_config = TransformerDatasetConfig.model_validate_json(artifacts.config_path.read_text())
    loaded_labels = LabelMapping.model_validate_json(artifacts.label_mapping_path.read_text())
    loaded_policy = TokenizerPolicy.model_validate_json(artifacts.tokenizer_policy_path.read_text())
    loaded_analysis = TokenLengthAnalysis.model_validate_json(artifacts.token_analysis_path.read_text())
    loaded_manifest = load_transformer_manifest(artifacts.manifest_path)
    loaded = TransformerDatasets(
        train=load_transformer_records(artifacts.train_path),
        validation=load_transformer_records(artifacts.validation_path),
        test=load_transformer_records(artifacts.test_path),
    )
    if (loaded_config, loaded_labels, loaded_policy, loaded_analysis, loaded_manifest) != (
        config, labels, policy, analysis, manifest
    ):
        raise TransformerArtifactPersistenceError("reloaded Transformer metadata differs")
    if loaded != datasets:
        raise TransformerArtifactPersistenceError("reloaded tokenized datasets differ")
    validate_dataset_alignment(splits, loaded, loaded_config, loaded_labels)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def _atomic_jsonl(path: Path, records: list[TransformerRecord]) -> None:
    content = "".join(
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_text(path, content)


def _atomic_text(path: Path, content: str) -> None:
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

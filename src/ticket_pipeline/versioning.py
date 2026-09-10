"""Deterministic local dataset fingerprinting and manifest generation."""

import hashlib
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel

from ticket_pipeline.language import (
    SUPPORTED_LANGUAGE_NAMES,
    LanguageDetectionConfig,
)
from ticket_pipeline.leakage import LeakageConfig
from ticket_pipeline.models import (
    DatasetFingerprints,
    DatasetManifest,
    DatasetManifestComparison,
    DatasetRecordCounts,
    DatasetSourceManifest,
    LanguageDetectionConfigSnapshot,
    LeakageConfigSnapshot,
    ProcessingConfigSnapshot,
    SemanticDeduplicationConfigSnapshot,
    SourceFileManifest,
)
from ticket_pipeline.semantic_deduplication import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
    DEFAULT_TOP_K,
)


PACKAGE_NAME = "support-ticket-ingestion-pipeline"
DATASET_VERSION_PREFIX = "ds_"
DATASET_VERSION_HASH_LENGTH = 12
_FILE_READ_CHUNK_SIZE = 1024 * 1024
_RECORD_HASH_DOMAIN = b"ticket-pipeline-records-v1\0"


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 hexadecimal digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def _canonicalize(value: object) -> object:
    """Convert supported values into a deterministic JSON-native structure."""

    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be canonically serialized")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetimes must include a timezone")
        utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("times must include a timezone")
        return value.isoformat(timespec="microseconds")
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(
            {
                item.name: getattr(value, item.name)
                for item in fields(value)
            }
        )
    if isinstance(value, Mapping):
        canonical_mapping: dict[str, object] = {}
        for key, item_value in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in canonical_mapping:
                raise ValueError("mapping keys collide after Unicode normalization")
            canonical_mapping[canonical_key] = _canonicalize(item_value)
        return canonical_mapping
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(canonical_items, key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise TypeError(
        f"unsupported canonical serialization type: {type(value).__name__}"
    )


def canonical_json(value: object) -> str:
    """Serialize supported values as stable, compact, Unicode JSON."""

    canonical_value = _canonicalize(value)
    return json.dumps(
        canonical_value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return UTF-8 bytes for canonical JSON serialization."""

    return canonical_json(value).encode("utf-8")


def fingerprint_value(value: object) -> str:
    """Return a SHA-256 fingerprint of one canonically serialized value."""

    return sha256_hex(canonical_json_bytes(value))


def fingerprint_records(records: Iterable[object]) -> str:
    """Hash records in ingestion order using unambiguous length framing."""

    digest = hashlib.sha256()
    digest.update(_RECORD_HASH_DOMAIN)
    for record in records:
        record_bytes = canonical_json_bytes(record)
        digest.update(len(record_bytes).to_bytes(8, byteorder="big"))
        digest.update(record_bytes)
    return digest.hexdigest()


def fingerprint_source_file(
    path: str | Path,
    *,
    source_name: str | None = None,
) -> SourceFileManifest:
    """Hash actual file bytes without including a machine-specific path."""

    file_path = Path(path)
    digest = hashlib.sha256()
    size_bytes = 0
    with file_path.open("rb") as source_file:
        while chunk := source_file.read(_FILE_READ_CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)

    return SourceFileManifest(
        name=file_path.name if source_name is None else source_name,
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def fingerprint_source_files(
    paths: Iterable[str | Path],
) -> DatasetSourceManifest:
    """Hash source files and sort them by stable file name."""

    return DatasetSourceManifest(
        files=[fingerprint_source_file(path) for path in paths]
    )


def capture_processing_config(
    *,
    semantic_deduplication_enabled: bool,
    semantic_model_name: str = DEFAULT_EMBEDDING_MODEL,
    semantic_threshold: float = DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
    semantic_top_k: int = DEFAULT_TOP_K,
    semantic_batch_size: int = DEFAULT_BATCH_SIZE,
    language_config: LanguageDetectionConfig | None = None,
    leakage_config: LeakageConfig | None = None,
) -> ProcessingConfigSnapshot:
    """Capture the existing settings that materially affect processed output."""

    language_settings = language_config or LanguageDetectionConfig()
    leakage_settings = leakage_config or LeakageConfig()
    semantic_settings = None
    if semantic_deduplication_enabled:
        semantic_settings = SemanticDeduplicationConfigSnapshot(
            model_name=semantic_model_name,
            threshold=semantic_threshold,
            top_k=semantic_top_k,
            batch_size=semantic_batch_size,
        )

    return ProcessingConfigSnapshot(
        semantic_deduplication=semantic_settings,
        language_detection=LanguageDetectionConfigSnapshot(
            backend="lingua-language-detector",
            confidence_threshold=language_settings.confidence_threshold,
            min_alphabetic_characters=(
                language_settings.min_alphabetic_characters
            ),
            supported_languages=list(SUPPORTED_LANGUAGE_NAMES),
        ),
        leakage=LeakageConfigSnapshot(
            forbidden_fields=sorted(leakage_settings.forbidden_fields),
            target_fields=sorted(leakage_settings.target_fields),
            target_proxy_fields=sorted(
                leakage_settings.target_proxy_fields
            ),
            group_fields=sorted(leakage_settings.group_fields),
        ),
    )


def get_pipeline_version() -> str:
    """Return the installed package version declared by the project."""

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError as error:
        raise RuntimeError(
            "pipeline package version is unavailable; install the project or "
            "pass pipeline_version explicitly"
        ) from error


def create_dataset_manifest(
    *,
    source: DatasetSourceManifest,
    processing_config: ProcessingConfigSnapshot,
    input_record_count: int,
    processed_records: Sequence[object],
    pipeline_version: str | None = None,
    created_at: datetime | None = None,
) -> DatasetManifest:
    """Create a content-addressed manifest for ordered processed records."""

    resolved_pipeline_version = (
        get_pipeline_version()
        if pipeline_version is None
        else pipeline_version
    )
    input_fingerprint = fingerprint_value(source.files)
    config_fingerprint = fingerprint_value(processing_config)
    output_fingerprint = fingerprint_records(processed_records)
    fingerprints = DatasetFingerprints(
        input=input_fingerprint,
        config=config_fingerprint,
        output=output_fingerprint,
    )
    identity_fingerprint = fingerprint_value(
        {
            "fingerprints": fingerprints,
            "pipeline_version": resolved_pipeline_version,
        }
    )

    return DatasetManifest(
        dataset_version=(
            f"{DATASET_VERSION_PREFIX}"
            f"{identity_fingerprint[:DATASET_VERSION_HASH_LENGTH]}"
        ),
        created_at=created_at or datetime.now(timezone.utc),
        pipeline_version=resolved_pipeline_version,
        source=source,
        counts=DatasetRecordCounts(
            input=input_record_count,
            processed=len(processed_records),
        ),
        processing_config=processing_config,
        fingerprints=fingerprints,
    )


def compare_dataset_manifests(
    first: DatasetManifest,
    second: DatasetManifest,
) -> DatasetManifestComparison:
    """Report which material manifest dimensions changed."""

    return DatasetManifestComparison(
        same_dataset_version=first.dataset_version == second.dataset_version,
        input_changed=(
            first.fingerprints.input != second.fingerprints.input
        ),
        config_changed=(
            first.fingerprints.config != second.fingerprints.config
        ),
        output_changed=(
            first.fingerprints.output != second.fingerprints.output
        ),
        pipeline_version_changed=(
            first.pipeline_version != second.pipeline_version
        ),
        counts_changed=first.counts != second.counts,
    )


def dataset_manifest_to_json(manifest: DatasetManifest) -> str:
    """Serialize a manifest as stable, readable JSON without a trailing newline."""

    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )


def write_dataset_manifest(
    manifest: DatasetManifest,
    path: str | Path,
) -> None:
    """Write a readable manifest with stable key ordering."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{dataset_manifest_to_json(manifest)}\n", encoding="utf-8"
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate a previously written dataset manifest."""

    with Path(path).open(encoding="utf-8") as manifest_file:
        return DatasetManifest.model_validate(json.load(manifest_file))

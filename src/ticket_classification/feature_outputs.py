"""Persistence and traceability for reusable sparse TF-IDF artifacts."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from ticket_classification.features import (
    CLASSIFICATION_INPUT_FIELDS,
    CLASSIFICATION_INPUT_POLICY,
    CLASSIFICATION_TEXT_SEPARATOR,
    FrozenSplits,
    TfidfConfig,
    TfidfFeatureResult,
)
from ticket_pipeline.versioning import fingerprint_value


VECTORIZER_FILENAME = "vectorizer.joblib"
TRAIN_FEATURES_FILENAME = "train_features.npz"
VALIDATION_FEATURES_FILENAME = "validation_features.npz"
TEST_FEATURES_FILENAME = "test_features.npz"
FEATURE_CONFIG_FILENAME = "feature_config.json"
FEATURE_METADATA_FILENAME = "feature_metadata.json"


class FeatureArtifactPersistenceError(RuntimeError):
    """TF-IDF artifacts could not be persisted or verified."""


class FeatureConfigArtifact(BaseModel):
    """Deterministic definition of the generated TF-IDF feature space."""

    model_config = ConfigDict(extra="forbid")

    feature_type: Literal["tfidf"] = "tfidf"
    input_policy: Literal["subject + message"] = CLASSIFICATION_INPUT_POLICY
    input_fields: list[StrictStr] = Field(
        default_factory=lambda: list(CLASSIFICATION_INPUT_FIELDS)
    )
    separator: Literal["\n\n"] = CLASSIFICATION_TEXT_SEPARATOR
    analyzer: Literal["word"]
    ngram_range: tuple[StrictInt, StrictInt]
    lowercase: StrictBool
    min_df: StrictInt | StrictFloat
    max_df: StrictInt | StrictFloat | None
    sublinear_tf: StrictBool
    max_features: StrictInt | None


class FeatureMetadata(BaseModel):
    """Source identity and measured sparse-matrix properties."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: StrictStr
    split_version: StrictStr
    feature_version: StrictStr
    feature_type: Literal["tfidf"] = "tfidf"
    fit_split: Literal["train"] = "train"
    vocabulary_size: int = Field(ge=1)
    feature_count: int = Field(ge=1)
    matrix_shapes: dict[StrictStr, tuple[int, int]]
    non_zero_counts: dict[StrictStr, int]
    matrix_format: Literal["csr"] = "csr"


@dataclass(frozen=True, slots=True)
class FeatureArtifacts:
    """Paths for one persisted and verified TF-IDF feature set."""

    vectorizer_path: Path
    train_features_path: Path
    validation_features_path: Path
    test_features_path: Path
    config_path: Path
    metadata_path: Path


def build_feature_config(config: TfidfConfig) -> FeatureConfigArtifact:
    """Capture every setting that defines the TF-IDF feature space."""

    return FeatureConfigArtifact(**config.model_dump())


def build_feature_metadata(
    splits: FrozenSplits,
    result: TfidfFeatureResult,
    config_artifact: FeatureConfigArtifact,
) -> FeatureMetadata:
    """Build deterministic source and sparse-matrix metadata."""

    split_version = f"split_{fingerprint_value(splits.manifest.model_dump(mode='json'))[:12]}"
    feature_identity = {
        "dataset_version": splits.manifest.dataset_version,
        "split_version": split_version,
        "configuration": config_artifact.model_dump(mode="json"),
    }
    feature_count = result.train.shape[1]
    if feature_count != len(result.vectorizer.vocabulary_):
        raise ValueError("feature count must equal fitted vocabulary size")
    matrices = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    if len({matrix.shape[1] for matrix in matrices.values()}) != 1:
        raise ValueError("all split matrices must share one feature space")
    return FeatureMetadata(
        dataset_version=splits.manifest.dataset_version,
        split_version=split_version,
        feature_version=f"tfidf_{fingerprint_value(feature_identity)[:12]}",
        vocabulary_size=len(result.vectorizer.vocabulary_),
        feature_count=feature_count,
        matrix_shapes={name: matrix.shape for name, matrix in matrices.items()},
        non_zero_counts={name: matrix.nnz for name, matrix in matrices.items()},
    )


def write_feature_artifacts(
    splits: FrozenSplits,
    result: TfidfFeatureResult,
    config: TfidfConfig,
    output_dir: str | Path,
) -> FeatureArtifacts:
    """Atomically persist and read back the fitted vectorizer and matrices."""

    target_dir = Path(output_dir)
    artifacts = FeatureArtifacts(
        vectorizer_path=target_dir / VECTORIZER_FILENAME,
        train_features_path=target_dir / TRAIN_FEATURES_FILENAME,
        validation_features_path=target_dir / VALIDATION_FEATURES_FILENAME,
        test_features_path=target_dir / TEST_FEATURES_FILENAME,
        config_path=target_dir / FEATURE_CONFIG_FILENAME,
        metadata_path=target_dir / FEATURE_METADATA_FILENAME,
    )
    config_artifact = build_feature_config(config)
    metadata = build_feature_metadata(splits, result, config_artifact)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_joblib_dump(artifacts.vectorizer_path, result.vectorizer)
        _atomic_sparse_save(artifacts.train_features_path, result.train)
        _atomic_sparse_save(artifacts.validation_features_path, result.validation)
        _atomic_sparse_save(artifacts.test_features_path, result.test)
        _atomic_write_json(artifacts.config_path, config_artifact)
        _atomic_write_json(artifacts.metadata_path, metadata)
        _verify_feature_artifacts(artifacts, result, config_artifact, metadata)
    except FeatureArtifactPersistenceError:
        raise
    except Exception as error:
        raise FeatureArtifactPersistenceError(
            "Failed to persist and verify TF-IDF feature artifacts."
        ) from error
    return artifacts


def load_vectorizer(path: str | Path) -> TfidfVectorizer:
    """Load and type-check a fitted TF-IDF vectorizer."""

    vectorizer = joblib.load(path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise FeatureArtifactPersistenceError("artifact is not a TF-IDF vectorizer")
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise FeatureArtifactPersistenceError("TF-IDF vectorizer is not fitted")
    return vectorizer


def load_sparse_features(path: str | Path) -> csr_matrix:
    """Load a persisted sparse matrix in canonical CSR format."""

    return load_npz(path).tocsr()


def load_feature_config(path: str | Path) -> FeatureConfigArtifact:
    return FeatureConfigArtifact.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_feature_metadata(path: str | Path) -> FeatureMetadata:
    return FeatureMetadata.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _verify_feature_artifacts(
    artifacts: FeatureArtifacts,
    result: TfidfFeatureResult,
    config: FeatureConfigArtifact,
    metadata: FeatureMetadata,
) -> None:
    loaded_vectorizer = load_vectorizer(artifacts.vectorizer_path)
    if loaded_vectorizer.vocabulary_ != result.vectorizer.vocabulary_:
        raise FeatureArtifactPersistenceError("reloaded vocabulary does not match")
    if not (loaded_vectorizer.idf_ == result.vectorizer.idf_).all():
        raise FeatureArtifactPersistenceError("reloaded IDF values do not match")
    loaded_matrices = {
        "train": load_sparse_features(artifacts.train_features_path),
        "validation": load_sparse_features(artifacts.validation_features_path),
        "test": load_sparse_features(artifacts.test_features_path),
    }
    expected_matrices = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    for name, loaded in loaded_matrices.items():
        expected = expected_matrices[name]
        if loaded.shape != expected.shape or loaded.nnz != expected.nnz:
            raise FeatureArtifactPersistenceError(f"reloaded {name} matrix differs")
        if (loaded != expected).nnz:
            raise FeatureArtifactPersistenceError(f"reloaded {name} values differ")
    if load_feature_config(artifacts.config_path) != config:
        raise FeatureArtifactPersistenceError("reloaded feature config differs")
    if load_feature_metadata(artifacts.metadata_path) != metadata:
        raise FeatureArtifactPersistenceError("reloaded feature metadata differs")


def _atomic_joblib_dump(path: Path, value: object) -> None:
    temporary = _temporary_path(path)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_sparse_save(path: Path, matrix: csr_matrix) -> None:
    temporary = _temporary_path(path)
    try:
        save_npz(temporary, matrix, compressed=True)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: BaseModel) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(
                value.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)

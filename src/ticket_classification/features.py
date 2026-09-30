"""Train-only TF-IDF feature construction for frozen classification splits."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ticket_classification.models import ClassificationRecord
from ticket_classification.split_outputs import (
    SPLIT_MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    SplitManifest,
    load_classification_records,
    load_split_manifest,
)


CLASSIFICATION_INPUT_FIELDS = ("subject", "message")
CLASSIFICATION_INPUT_POLICY = "subject + message"
CLASSIFICATION_TEXT_SEPARATOR = "\n\n"


class TfidfConfig(BaseModel):
    """Frozen word-level TF-IDF baseline configuration."""

    model_config = ConfigDict(extra="forbid")

    analyzer: Literal["word"] = "word"
    ngram_range: tuple[StrictInt, StrictInt] = (1, 2)
    lowercase: StrictBool = True
    min_df: StrictInt | StrictFloat = Field(default=2)
    max_df: StrictInt | StrictFloat | None = None
    sublinear_tf: StrictBool = False
    max_features: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_document_frequency_and_ngrams(self) -> "TfidfConfig":
        lower, upper = self.ngram_range
        if lower < 1:
            raise ValueError("ngram lower bound must be at least 1")
        if upper < lower:
            raise ValueError("ngram upper bound must be at least the lower bound")
        _validate_document_frequency("min_df", self.min_df)
        if self.max_df is not None:
            _validate_document_frequency("max_df", self.max_df)
            if type(self.min_df) is type(self.max_df) and self.max_df < self.min_df:
                raise ValueError("max_df must not be less than min_df")
        return self


@dataclass(frozen=True, slots=True)
class FrozenSplits:
    """Phase 2.2 records loaded in their persisted row order."""

    train: list[ClassificationRecord]
    validation: list[ClassificationRecord]
    test: list[ClassificationRecord]
    manifest: SplitManifest


@dataclass(frozen=True, slots=True)
class TfidfFeatureResult:
    """One fitted vectorizer and its aligned sparse split matrices."""

    vectorizer: TfidfVectorizer
    train: csr_matrix
    validation: csr_matrix
    test: csr_matrix


def build_classification_text(record: ClassificationRecord) -> str:
    """Return the Phase 2.1 frozen ``subject + message`` representation.

    Phase 2.1 has already stripped empty fields and joined non-empty subject
    and message values with two newlines. The split record intentionally does
    not retain those raw fields, so Phase 2.3 validates the source-field policy
    and reuses its deterministic text without alteration.
    """

    if tuple(record.source_fields) != CLASSIFICATION_INPUT_FIELDS:
        raise ValueError(
            "classification record must use source_fields "
            f"{list(CLASSIFICATION_INPUT_FIELDS)!r}"
        )
    return record.text


def load_frozen_splits(split_dir: str | Path) -> FrozenSplits:
    """Load and verify Phase 2.2 artifacts without regenerating the split."""

    directory = Path(split_dir)
    manifest = load_split_manifest(directory / SPLIT_MANIFEST_FILENAME)
    groups = {
        "train": load_classification_records(directory / TRAIN_FILENAME),
        "validation": load_classification_records(directory / VALIDATION_FILENAME),
        "test": load_classification_records(directory / TEST_FILENAME),
    }
    expected_ids = {
        "train": manifest.train_record_ids,
        "validation": manifest.validation_record_ids,
        "test": manifest.test_record_ids,
    }
    expected_counts = {
        "train": manifest.train_count,
        "validation": manifest.validation_count,
        "test": manifest.test_count,
    }
    for name, records in groups.items():
        actual_ids = [record.record_id for record in records]
        if len(records) != expected_counts[name]:
            raise ValueError(
                f"{name} split count does not match manifest: "
                f"expected={expected_counts[name]}, actual={len(records)}"
            )
        if actual_ids != expected_ids[name]:
            raise ValueError(f"{name} split record order does not match manifest")

    return FrozenSplits(manifest=manifest, **groups)


def fit_tfidf(
    train_records: list[ClassificationRecord],
    config: TfidfConfig | None = None,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit exactly one TF-IDF vocabulary and IDF model on training records."""

    if not train_records:
        raise ValueError("train_records must not be empty")
    settings = config or TfidfConfig()
    # ``None`` means "use scikit-learn's default/unset behavior" in our
    # persisted contract. Scikit-learn itself expects those kwargs omitted.
    vectorizer = TfidfVectorizer(**settings.model_dump(exclude_none=True))
    matrix = vectorizer.fit_transform(
        [build_classification_text(record) for record in train_records]
    ).tocsr()
    _validate_sparse_matrix(matrix, len(train_records), len(vectorizer.vocabulary_))
    return vectorizer, matrix


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    records: list[ClassificationRecord],
) -> csr_matrix:
    """Transform records without changing the fitted TF-IDF feature space."""

    matrix = vectorizer.transform(
        [build_classification_text(record) for record in records]
    ).tocsr()
    _validate_sparse_matrix(matrix, len(records), len(vectorizer.vocabulary_))
    return matrix


def build_tfidf_features(
    splits: FrozenSplits,
    config: TfidfConfig | None = None,
) -> TfidfFeatureResult:
    """Fit on train, then transform validation and test in frozen row order."""

    vectorizer, train_matrix = fit_tfidf(splits.train, config)
    validation_matrix = transform_tfidf(vectorizer, splits.validation)
    test_matrix = transform_tfidf(vectorizer, splits.test)
    return TfidfFeatureResult(
        vectorizer=vectorizer,
        train=train_matrix,
        validation=validation_matrix,
        test=test_matrix,
    )


def _validate_document_frequency(name: str, value: int | float) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer count or float proportion")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"integer {name} must be at least 1")
        return
    if not 0.0 < value <= 1.0:
        raise ValueError(f"float {name} must be in the interval (0.0, 1.0]")


def _validate_sparse_matrix(matrix: csr_matrix, rows: int, columns: int) -> None:
    if not issparse(matrix) or not isinstance(matrix, csr_matrix):
        raise TypeError("TF-IDF output must be a CSR sparse matrix")
    if matrix.shape != (rows, columns):
        raise ValueError(
            "TF-IDF matrix shape is inconsistent with records and vocabulary: "
            f"expected={(rows, columns)}, actual={matrix.shape}"
        )

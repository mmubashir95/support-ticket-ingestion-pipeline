"""Small public workflow for running the classification dataset audit."""

from pathlib import Path

from ticket_classification.classical_outputs import run_logistic_regression_baseline
from ticket_classification.audit import audit_classification_dataset
from ticket_classification.dataset import (
    ClassificationAuditConfig,
    build_classification_records,
    load_accepted_records,
    load_optional_dataset_manifest,
)
from ticket_classification.models import ClassificationDatasetAudit
from ticket_classification.feature_outputs import (
    FeatureArtifacts,
    write_feature_artifacts,
)
from ticket_classification.features import (
    TfidfConfig,
    build_tfidf_features,
    load_frozen_splits,
)
from ticket_classification.outputs import (
    ClassificationAuditArtifacts,
    write_classification_audit_outputs,
)
from ticket_classification.split_outputs import SplitArtifacts, write_split_artifacts
from ticket_classification.splitting import (
    SplitConfig,
    split_classification_records,
)


def run_classification_dataset_audit(
    accepted_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    config: ClassificationAuditConfig | None = None,
) -> ClassificationDatasetAudit | tuple[ClassificationDatasetAudit, ClassificationAuditArtifacts]:
    """Audit an accepted Phase 1 JSONL dataset and optionally persist reports."""

    tickets = load_accepted_records(accepted_path)
    manifest = load_optional_dataset_manifest(manifest_path)
    audit = audit_classification_dataset(
        tickets,
        config,
        dataset_manifest=manifest,
    )
    if output_dir is None:
        return audit
    return audit, write_classification_audit_outputs(audit, output_dir)


def run_classification_dataset_split(
    accepted_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: SplitConfig | None = None,
) -> SplitArtifacts:
    """Build, validate, and freeze the reusable Phase 2 classification split.

    The authoritative dataset version comes from the Phase 1 manifest. When a
    caller supplies a configuration, integrity validation rejects any version
    mismatch before artifacts are written.
    """

    tickets = load_accepted_records(accepted_path)
    manifest = load_optional_dataset_manifest(manifest_path)
    if manifest is None:  # pragma: no cover - the required path always loads or fails
        raise ValueError("manifest_path is required")
    settings = config or SplitConfig(dataset_version=manifest.dataset_version)
    records = build_classification_records(
        tickets,
        ClassificationAuditConfig(target_field=settings.target_field),
    )
    result = split_classification_records(records, settings)
    return write_split_artifacts(
        records,
        result,
        settings,
        output_dir,
        expected_dataset_version=manifest.dataset_version,
    )


def run_tfidf_feature_pipeline(
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TfidfConfig | None = None,
) -> FeatureArtifacts:
    """Build TF-IDF artifacts from the existing split without resplitting."""

    settings = config or TfidfConfig()
    splits = load_frozen_splits(split_dir)
    result = build_tfidf_features(splits, settings)
    return write_feature_artifacts(splits, result, settings, output_dir)

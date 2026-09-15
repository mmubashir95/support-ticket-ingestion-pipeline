"""Small public workflow for running the classification dataset audit."""

from pathlib import Path

from ticket_classification.audit import audit_classification_dataset
from ticket_classification.dataset import (
    ClassificationAuditConfig,
    load_accepted_records,
    load_optional_dataset_manifest,
)
from ticket_classification.models import ClassificationDatasetAudit
from ticket_classification.outputs import (
    ClassificationAuditArtifacts,
    write_classification_audit_outputs,
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

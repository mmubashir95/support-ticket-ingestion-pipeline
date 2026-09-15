"""Persistence for classification dataset audit artifacts."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ticket_classification.models import ClassificationDatasetAudit


CLASSIFICATION_AUDIT_JSON_FILENAME = "classification_dataset_audit.json"
CLASSIFICATION_AUDIT_MARKDOWN_FILENAME = "classification_dataset_audit.md"


class ClassificationAuditPersistenceError(RuntimeError):
    """Fatal failure while writing classification audit artifacts."""


@dataclass(frozen=True, slots=True)
class ClassificationAuditArtifacts:
    """Paths written for one classification dataset audit."""

    json_path: Path
    markdown_path: Path


def classification_audit_to_json(audit: ClassificationDatasetAudit) -> str:
    """Serialize an audit deterministically as readable JSON."""

    return json.dumps(
        audit.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )


def classification_audit_to_markdown(audit: ClassificationDatasetAudit) -> str:
    """Render a concise human-readable classification audit report."""

    lines = [
        "# Classification Dataset Audit",
        "",
        "## Dataset",
        "",
        f"- Dataset version: {audit.dataset_summary.dataset_version or 'not provided'}",
        f"- Total records: {audit.dataset_summary.total_records}",
        f"- Usable labeled records: {audit.dataset_summary.usable_labeled_records}",
        f"- Readiness: {audit.dataset_readiness.value}",
        "",
        "## Classification Task",
        "",
        f"- Target field: `{audit.classification_task.target_field}`",
        f"- Task type: `{audit.classification_task.task_type}`",
        f"- Input fields: {', '.join(f'`{field}`' for field in audit.classification_task.input_fields)}",
        "",
        "## Target Summary",
        "",
        f"- Unique classes: {audit.target_summary.unique_class_count}",
        f"- Missing labels: {audit.target_summary.records_with_missing_labels}",
        f"- Blank labels: {audit.target_summary.records_with_blank_labels}",
        f"- Invalid labels: {audit.target_summary.records_with_invalid_labels}",
        "",
        "## Class Distribution",
        "",
        "| Class | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]
    for item in audit.class_distribution.classes:
        lines.append(f"| {item.class_name} | {item.count} | {item.percentage:.6f} |")

    lines.extend(
        [
            "",
            f"- Largest class: {audit.class_distribution.largest_class or 'not available'}",
            f"- Smallest class: {audit.class_distribution.smallest_class or 'not available'}",
            (
                "- Largest-to-smallest ratio: "
                f"{audit.class_distribution.largest_to_smallest_ratio}"
            ),
            f"- Imbalance severity: {audit.class_distribution.imbalance_severity}",
            "",
            "## Rare Classes",
            "",
        ]
    )
    if audit.rare_classes:
        for item in audit.rare_classes:
            lines.append(f"- {item.class_name}: {item.count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Text Length",
            "",
            f"- Character length mean: {audit.text_length_summary.character_length.mean}",
            f"- Character length P95: {audit.text_length_summary.character_length.p95}",
            f"- Word length mean: {audit.text_length_summary.word_length.mean}",
            f"- Word length P95: {audit.text_length_summary.word_length.p95}",
            f"- Empty inputs: {audit.text_length_summary.empty_input_records}",
            f"- Very short inputs: {audit.text_length_summary.short_input_records}",
            f"- Placeholder-only inputs: {audit.text_length_summary.placeholder_only_records}",
            "",
            "## Duplicate Inputs",
            "",
            f"- Duplicated input groups: {audit.conflicting_labels.duplicated_input_groups}",
            f"- Consistent duplicate groups: {audit.conflicting_labels.groups_with_consistent_labels}",
            f"- Conflicting duplicate groups: {audit.conflicting_labels.groups_with_conflicting_labels}",
            "",
            "## Input Field Policy",
            "",
            "| Field | Usage | Reason |",
            "| --- | --- | --- |",
        ]
    )
    for item in audit.input_field_policy:
        lines.append(f"| `{item.field_name}` | {item.usage.value} | {item.reason} |")

    lines.extend(["", "## Leakage Risks", ""])
    if audit.leakage_risks:
        for item in audit.leakage_risks:
            lines.append(f"- `{item.field_name}`: {item.reason}")
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if audit.warnings:
        for warning in audit.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Recommended Next Production Step",
            "",
            "Create and freeze the train / validation / test dataset split.",
            "",
        ]
    )
    return "\n".join(lines)


def write_classification_audit_outputs(
    audit: ClassificationDatasetAudit,
    output_dir: str | Path,
) -> ClassificationAuditArtifacts:
    """Persist structured and human-readable classification audit artifacts."""

    target_dir = Path(output_dir)
    artifacts = ClassificationAuditArtifacts(
        json_path=target_dir / CLASSIFICATION_AUDIT_JSON_FILENAME,
        markdown_path=target_dir / CLASSIFICATION_AUDIT_MARKDOWN_FILENAME,
    )
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            artifacts.json_path,
            f"{classification_audit_to_json(audit)}\n",
        )
        _atomic_write_text(
            artifacts.markdown_path,
            classification_audit_to_markdown(audit),
        )
    except ClassificationAuditPersistenceError:
        raise
    except Exception as error:
        raise ClassificationAuditPersistenceError(
            "Failed to persist classification audit artifacts."
        ) from error
    return artifacts


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception as error:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ClassificationAuditPersistenceError(
            f"Failed to write classification audit artifact: {path.name}"
        ) from error


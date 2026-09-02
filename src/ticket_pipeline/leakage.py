"""Policy-based leakage checks for future model feature and split safety."""

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from ticket_pipeline.deduplication import ExactDuplicateResult
from ticket_pipeline.models import (
    GroupIdentifierValue,
    LeakageCheckMetadata,
    LeakageCheckStatus,
    LeakageGroupingMetadata,
    LeakageIssue,
    LeakageIssueType,
    LeakageTargetMetadata,
    Ticket,
)
from ticket_pipeline.semantic_deduplication import SemanticDuplicateResult


DEFAULT_FORBIDDEN_FIELDS = frozenset({"answer"})
logger = logging.getLogger(__name__)


def _normalize_field_names(value: object, setting_name: str) -> frozenset[str]:
    if isinstance(value, str) or not isinstance(value, Collection):
        raise ValueError(f"{setting_name} must be a collection of field names")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{setting_name} must contain non-blank strings")
    return frozenset(value)


@dataclass(frozen=True, slots=True)
class LeakageConfig:
    """Explicit prediction-time field and grouping policy."""

    forbidden_fields: Collection[str] = field(
        default_factory=lambda: DEFAULT_FORBIDDEN_FIELDS
    )
    target_fields: Collection[str] = field(default_factory=frozenset)
    target_proxy_fields: Collection[str] = field(default_factory=frozenset)
    group_fields: Collection[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for setting_name in (
            "forbidden_fields",
            "target_fields",
            "target_proxy_fields",
            "group_fields",
        ):
            normalized = _normalize_field_names(
                getattr(self, setting_name),
                setting_name,
            )
            object.__setattr__(self, setting_name, normalized)


def is_populated(value: object) -> bool:
    """Return whether a field contains meaningful information.

    False and numeric zero are meaningful. Blank strings and empty built-in
    containers are not.
    """

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return bool(value)
    return True


class LeakageChecker:
    """Inspect configured fields and reuse existing duplicate relationships."""

    def __init__(self, config: LeakageConfig | None = None) -> None:
        self.config = config or LeakageConfig()

    def check(
        self,
        record_fields: Mapping[str, object],
        *,
        exact_duplicate: ExactDuplicateResult | None = None,
        semantic_duplicate: SemanticDuplicateResult | None = None,
    ) -> LeakageCheckMetadata:
        """Return leakage metadata without raising for record-level failures."""

        try:
            return self._check(
                record_fields,
                exact_duplicate=exact_duplicate,
                semantic_duplicate=semantic_duplicate,
            )
        except Exception:
            logger.exception("Leakage check failed")
            return LeakageCheckMetadata(status=LeakageCheckStatus.FAILED)

    def _check(
        self,
        record_fields: Mapping[str, object],
        *,
        exact_duplicate: ExactDuplicateResult | None,
        semantic_duplicate: SemanticDuplicateResult | None,
    ) -> LeakageCheckMetadata:
        issues = self._field_issues(
            record_fields,
            self.config.forbidden_fields,
            LeakageIssueType.FUTURE_FIELD,
        )

        # A populated target/proxy field is the normal, expected shape of a
        # labeled training record, not evidence that this record is invalid,
        # so it is recorded as informational metadata rather than a warning.
        checked_field_sets = [
            self.config.forbidden_fields,
            self.config.target_fields,
            self.config.group_fields,
        ]
        target_proxy_fields_present: list[str] = []
        if self.config.target_fields:
            target_proxy_fields_present = self._populated_fields(
                record_fields, self.config.target_proxy_fields
            )
            checked_field_sets.append(self.config.target_proxy_fields)
        targets = LeakageTargetMetadata(
            target_fields=self._populated_fields(
                record_fields, self.config.target_fields
            ),
            target_proxy_fields=target_proxy_fields_present,
        )

        grouping = LeakageGroupingMetadata(
            exact_duplicate_group_id=self._exact_group(exact_duplicate),
            semantic_duplicate_group_id=self._semantic_group(
                semantic_duplicate
            ),
            identifiers=self._group_identifiers(record_fields),
        )
        self._validate_duplicate_alignment(exact_duplicate, semantic_duplicate)

        unchecked_fields = self._unchecked_fields(
            record_fields, *checked_field_sets
        )

        status = (
            LeakageCheckStatus.WARNING
            if issues
            else LeakageCheckStatus.CLEAN
        )
        return LeakageCheckMetadata(
            status=status,
            issues=issues,
            targets=targets,
            grouping=grouping,
            unchecked_fields=unchecked_fields,
        )

    @staticmethod
    def _populated_fields(
        record_fields: Mapping[str, object],
        configured_fields: Collection[str],
    ) -> list[str]:
        return [
            field_name
            for field_name in sorted(configured_fields)
            if field_name in record_fields
            and is_populated(record_fields[field_name])
        ]

    @staticmethod
    def _field_issues(
        record_fields: Mapping[str, object],
        configured_fields: Collection[str],
        issue_type: LeakageIssueType,
    ) -> list[LeakageIssue]:
        return [
            LeakageIssue(type=issue_type, field=field_name)
            for field_name in LeakageChecker._populated_fields(
                record_fields, configured_fields
            )
        ]

    @staticmethod
    def _unchecked_fields(
        record_fields: Mapping[str, object],
        *configured_field_sets: Collection[str],
    ) -> list[str]:
        """Report configured field names the checker never saw any value for.

        A missing configured field is always treated as safe (never a
        warning), but silently missing a field is different from having
        confirmed it is genuinely empty: a caller that forgets to pass
        ``source_fields`` to ``annotate_ticket_leakage`` cannot have any
        source-only forbidden field (for example this project's default
        ``answer``) show up here, which would otherwise be indistinguishable
        from a record that was checked and found clean.
        """

        all_configured: set[str] = set()
        for configured_fields in configured_field_sets:
            all_configured.update(configured_fields)
        return sorted(name for name in all_configured if name not in record_fields)

    def _group_identifiers(
        self,
        record_fields: Mapping[str, object],
    ) -> dict[str, GroupIdentifierValue]:
        identifiers: dict[str, GroupIdentifierValue] = {}
        for field_name in sorted(self.config.group_fields):
            if field_name not in record_fields:
                continue
            value = record_fields[field_name]
            if not is_populated(value):
                continue
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError("group identifiers must be scalar values")
            identifiers[field_name] = value
        return identifiers

    @staticmethod
    def _exact_group(
        result: ExactDuplicateResult | None,
    ) -> int | None:
        if result is None:
            return None
        is_duplicate = result["is_exact_duplicate"]
        duplicate_of = result["duplicate_of"]
        record_index = result["record_index"]
        if not isinstance(is_duplicate, bool):
            raise ValueError("invalid exact duplicate result")
        if is_duplicate:
            if (
                isinstance(duplicate_of, bool)
                or not isinstance(duplicate_of, int)
                or duplicate_of < 0
                or duplicate_of >= record_index
            ):
                raise ValueError("invalid exact duplicate link")
            return duplicate_of
        if duplicate_of is not None:
            raise ValueError("non-duplicate record cannot have an exact link")
        return None

    @staticmethod
    def _semantic_group(
        result: SemanticDuplicateResult | None,
    ) -> int | None:
        if result is None:
            return None
        is_candidate = result["is_semantic_duplicate_candidate"]
        duplicate_of = result["candidate_duplicate_of"]
        record_index = result["record_index"]
        if not isinstance(is_candidate, bool):
            raise ValueError("invalid semantic duplicate result")
        if is_candidate:
            if (
                isinstance(duplicate_of, bool)
                or not isinstance(duplicate_of, int)
                or duplicate_of < 0
                or duplicate_of >= record_index
            ):
                raise ValueError("invalid semantic duplicate link")
            return duplicate_of
        if duplicate_of is not None:
            raise ValueError("non-candidate record cannot have a semantic link")
        return None

    @staticmethod
    def _validate_duplicate_alignment(
        exact_result: ExactDuplicateResult | None,
        semantic_result: SemanticDuplicateResult | None,
    ) -> None:
        if (
            exact_result is not None
            and semantic_result is not None
            and exact_result["record_index"] != semantic_result["record_index"]
        ):
            raise ValueError("duplicate results must refer to the same record")


def annotate_ticket_leakage(
    ticket: Ticket,
    checker: LeakageChecker,
    *,
    source_fields: Mapping[str, object] | None = None,
    exact_duplicate: ExactDuplicateResult | None = None,
    semantic_duplicate: SemanticDuplicateResult | None = None,
) -> Ticket:
    """Return a ticket copy with leakage metadata and unchanged prior fields."""

    record_fields: dict[str, object] = ticket.model_dump(mode="python")
    if source_fields is not None:
        record_fields.update(source_fields)
    metadata = checker.check(
        record_fields,
        exact_duplicate=exact_duplicate,
        semantic_duplicate=semantic_duplicate,
    )
    return ticket.model_copy(update={"leakage_check": metadata})

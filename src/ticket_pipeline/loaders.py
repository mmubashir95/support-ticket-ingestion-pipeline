"""Load source ticket records and map them into canonical dictionaries."""

import csv
import json
import math
from pathlib import Path
from typing import Any


TAG_FIELDS = tuple(f"tag_{index}" for index in range(1, 9))


def map_source_record(record: dict[str, object]) -> dict[str, object]:
    """Map one source-format record to canonical ticket field names."""

    return {
        "subject": _blank_to_none(record.get("subject")),
        "message": record.get("body"),
        "ticket_type": record.get("type"),
        "queue": record.get("queue"),
        "priority": record.get("priority"),
        "language": record.get("language"),
        "source_version": _to_int_when_simple(record.get("version")),
        "tags": _collect_tags(record),
    }


def load_csv(path: str | Path) -> list[dict[str, object]]:
    """Load a CSV file and return canonical ticket dictionaries."""

    file_path = Path(path)
    with file_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [map_source_record(dict(row)) for row in reader]


def load_json(path: str | Path) -> list[dict[str, object]]:
    """Load a JSON list of source records and return canonical dictionaries."""

    file_path = Path(path)
    with file_path.open(encoding="utf-8") as json_file:
        data = json.load(json_file)

    if not isinstance(data, list):
        raise ValueError("JSON input must contain a top-level list of records.")

    records: list[dict[str, object]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"JSON record at index {index} must be an object.")
        records.append(map_source_record(item))

    return records


def load_records(path: str | Path) -> list[dict[str, object]]:
    """Load records from a supported source file based on file extension."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_csv(file_path)
    if suffix == ".json":
        return load_json(file_path)

    raise ValueError(f"Unsupported input file extension: {file_path.suffix}")


def _collect_tags(record: dict[str, object]) -> list[str]:
    tags: list[str] = []
    for field in TAG_FIELDS:
        value = record.get(field)
        if _is_blank_tag(value):
            continue
        tags.append(str(value))
    return tags


def _blank_to_none(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _to_int_when_simple(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return value


def _is_blank_tag(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped == "" or stripped.lower() == "nan"
    return False

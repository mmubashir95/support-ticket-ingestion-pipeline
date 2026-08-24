import json

import pytest

from ticket_pipeline.loaders import load_csv, load_json, load_records, map_source_record


def test_map_source_record_applies_canonical_field_names() -> None:
    mapped = map_source_record(
        {
            "subject": "Payment issue",
            "body": "My card was charged twice.",
            "answer": "Refund guidance sent.",
            "type": "Incident",
            "queue": "Billing and Payments",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "billing",
            "tag_2": "refund",
            "tag_3": "",
            "tag_4": None,
            "tag_5": "nan",
        }
    )

    assert mapped == {
        "subject": "Payment issue",
        "message": "My card was charged twice.",
        "ticket_type": "Incident",
        "queue": "Billing and Payments",
        "priority": "high",
        "language": "en",
        "source_version": 52,
        "tags": ["billing", "refund"],
    }


def test_load_csv_returns_canonical_records() -> None:
    records = load_csv("tests/fixtures/tickets_sample.csv")

    assert len(records) == 2
    assert records[0]["message"] == "My card was charged twice."
    assert records[0]["ticket_type"] == "Incident"
    assert records[0]["source_version"] == 52
    assert records[0]["tags"] == ["billing", "refund"]
    assert "answer" not in records[0]


def test_load_csv_omits_blank_tags_and_maps_blank_subject_to_none() -> None:
    records = load_csv("tests/fixtures/tickets_sample.csv")

    assert records[1]["subject"] is None
    assert records[1]["tags"] == []


def test_load_json_returns_canonical_records() -> None:
    records = load_json("tests/fixtures/tickets_sample.json")

    assert len(records) == 2
    assert records[0]["message"] == "My card was charged twice."
    assert records[0]["ticket_type"] == "Incident"
    assert records[0]["source_version"] == 52
    assert records[0]["tags"] == ["billing", "refund"]
    assert "answer" not in records[0]


def test_csv_and_json_mapping_match_for_same_logical_ticket() -> None:
    csv_record = load_csv("tests/fixtures/tickets_sample.csv")[0]
    json_record = load_json("tests/fixtures/tickets_sample.json")[0]

    assert csv_record == json_record


def test_load_records_selects_loader_by_extension() -> None:
    csv_records = load_records("tests/fixtures/tickets_sample.csv")
    json_records = load_records("tests/fixtures/tickets_sample.json")

    assert len(csv_records) == 2
    assert len(json_records) == 2


def test_load_records_fails_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_records("tests/fixtures/missing.csv")


def test_load_records_fails_for_unsupported_extension(tmp_path) -> None:
    source_file = tmp_path / "tickets.txt"
    source_file.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported input file extension"):
        load_records(source_file)


def test_load_json_fails_for_malformed_json(tmp_path) -> None:
    source_file = tmp_path / "broken.json"
    source_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_json(source_file)


def test_load_json_fails_when_root_is_not_list(tmp_path) -> None:
    source_file = tmp_path / "object.json"
    source_file.write_text('{"body": "Not a list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="top-level list"):
        load_json(source_file)


def test_load_json_fails_when_list_item_is_not_object(tmp_path) -> None:
    source_file = tmp_path / "invalid_item.json"
    source_file.write_text('[{"body": "Valid shape"}, "bad item"]', encoding="utf-8")

    with pytest.raises(ValueError, match="index 1"):
        load_json(source_file)


import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ticket_classification.benchmarking import (
    DEFAULT_RUN_ROOT,
    BenchmarkConfig,
    artifact_sizes,
    calculate_costs,
    deterministic_sample,
    make_batches,
    measure_batches,
    percentile,
    regenerate_report,
    render_report,
    validate_artifacts,
)


def test_configuration_validation():
    with pytest.raises(ValidationError):
        BenchmarkConfig(batch_sizes=(1, 1))
    with pytest.raises(ValidationError):
        BenchmarkConfig(repetitions=0)
    with pytest.raises(ValidationError):
        BenchmarkConfig(hourly_costs={"cpu": -1})


def test_default_paths_are_repository_anchored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = BenchmarkConfig()
    assert config.run_root == DEFAULT_RUN_ROOT
    assert config.run_root.is_absolute()
    assert config.output_dir == DEFAULT_RUN_ROOT / "inference_benchmark"


def test_percentiles_use_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 50)


def test_measurement_excludes_warmup_and_counts_actual_records():
    calls = []
    ticks = iter([0.0, 1.0, 1.0, 3.0, 3.0, 6.0, 6.0, 10.0])

    def infer(batch):
        calls.append(tuple(batch))
        return [0] * len(batch)

    result = measure_batches(
        [1, 2, 3], 2, infer, warmup_iterations=2, repetitions=2,
        clock=lambda: next(ticks),
    )
    assert len(calls) == 6
    assert result.batch_count == 4
    assert result.processed_tickets == 6
    assert result.durations_seconds == (1.0, 2.0, 3.0, 4.0)
    assert result.p50_batch_seconds == 2.5
    assert result.throughput_tickets_per_second == pytest.approx(0.6)


def test_measurement_rejects_empty_input_and_bad_iterations():
    with pytest.raises(ValueError, match="must not be empty"):
        measure_batches([], 1, lambda _: None, warmup_iterations=0, repetitions=1)
    with pytest.raises(ValueError, match="invalid"):
        measure_batches([1], 1, lambda _: None, warmup_iterations=-1, repetitions=1)


def test_batches_keep_short_final_batch():
    assert make_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_sampling_is_deterministic_and_source_ordered():
    first = deterministic_sample(list(range(20)), 6, 17)
    assert first == deterministic_sample(list(range(20)), 6, 17)
    assert first == sorted(first)
    assert first != deterministic_sample(list(range(20)), 6, 18)


def test_artifact_sizes_deduplicate_shared_files(tmp_path):
    one, two = tmp_path / "one.bin", tmp_path / "two.bin"
    one.write_bytes(b"123")
    two.write_bytes(b"45678")
    result = artifact_sizes((tmp_path, one))
    assert result["total_bytes"] == 8
    assert len(result["files"]) == 2


def test_cost_calculation_and_unavailable_cost():
    assert calculate_costs(100, None) is None
    cost = calculate_costs(100, 3.6)
    assert cost["cost_per_ticket"] == pytest.approx(0.00001)
    assert cost["cost_per_1000_tickets"] == pytest.approx(0.01)
    assert cost["cost_per_1000000_tickets"] == pytest.approx(10)


def _results_fixture():
    measurement = {
        "batch_size": 1, "p50_batch_seconds": 0.01, "p95_batch_seconds": 0.02,
        "throughput_tickets_per_second": 50.0, "amortized_seconds_per_ticket": 0.02,
        "processed_tickets": 10, "cost": None,
    }
    return {
        "schema_version": "phase_2_11_v1", "executed_at": "2026-01-01T00:00:00+00:00",
        "artifact_validation": {"selected_models": {"logistic_regression": {"weighting": "balanced"}}},
        "environment": {"python": "test"},
        "sample": {"dataset_version": "ds_test", "split": "test", "count": 2,
            "sampling_policy": "test policy", "random_seed": 42,
            "input_character_lengths": {"minimum": 1, "maximum": 2, "p50": 1.5, "p95": 1.95}},
        "results": [{"model": "logistic_regression", "device": "cpu", "timing_scope": "end-to-end",
            "measurements": [measurement], "weights_size": {"total_bytes": 3},
            "artifact_size": {"total_bytes": 8}, "memory": {"process_peak_rss_bytes": 100}}],
        "limitations": ["test limitation"],
    }


def test_report_generation_uses_structured_results(tmp_path):
    data = _results_fixture()
    text = render_report(data)
    assert "50.000" in text
    assert "test limitation" in text
    results = tmp_path / "benchmark_results.json"
    results.write_text(json.dumps(data), encoding="utf-8")
    target = regenerate_report(results)
    assert target.read_text(encoding="utf-8") == text


def test_current_archive_reports_selected_artifacts_missing():
    root = Path("artifacts/classification/run")
    if not root.exists():
        pytest.skip("repository artifacts are unavailable")
    config = BenchmarkConfig(run_root=root, models=("logistic_regression",),
                             include_available_accelerator=False)
    with pytest.raises(FileNotFoundError, match="weighted_model.joblib"):
        validate_artifacts(config)

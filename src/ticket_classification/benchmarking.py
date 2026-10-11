"""Phase 2.11 reproducible inference benchmarking utilities.

Model loading is deliberately strict: only the artifacts selected by the
Phase 2.9/2.10 policies are accepted and Hugging Face loading is local-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
from time import perf_counter
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from ticket_classification.calibration_workflow import verify_phase29_policy
from ticket_classification.classical_outputs import load_linear_svm, load_logistic_regression
from ticket_classification.feature_outputs import load_feature_metadata, load_vectorizer
from ticket_classification.features import build_classification_text, load_frozen_splits
from ticket_classification.split_outputs import _atomic_write_text


ModelName = Literal["logistic_regression", "linear_svm", "transformer"]
DeviceName = Literal["cpu", "cuda", "mps"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "artifacts" / "classification" / "run"


class BenchmarkConfig(BaseModel):
    """Validated configuration persisted with every benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_root: Path = DEFAULT_RUN_ROOT
    output_dir: Path = DEFAULT_RUN_ROOT / "inference_benchmark"
    models: tuple[ModelName, ...] = (
        "logistic_regression", "linear_svm", "transformer"
    )
    transformer_devices: tuple[DeviceName, ...] = ("cpu",)
    include_available_accelerator: bool = True
    batch_sizes: tuple[StrictInt, ...] = (1, 8, 16, 32, 64)
    warmup_iterations: StrictInt = Field(default=5, ge=0)
    repetitions: StrictInt = Field(default=30, ge=1)
    sample_count: StrictInt = Field(default=512, ge=1)
    random_seed: StrictInt = Field(default=42, ge=0, le=4294967295)
    split: Literal["validation", "test"] = "test"
    hourly_costs: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_collections(self):
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be nonempty and unique")
        if not self.batch_sizes or len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must be nonempty and unique")
        if any(value < 1 for value in self.batch_sizes):
            raise ValueError("batch sizes must be positive")
        if len(set(self.transformer_devices)) != len(self.transformer_devices):
            raise ValueError("transformer_devices must be unique")
        if any(not math.isfinite(rate) or rate < 0 for rate in self.hourly_costs.values()):
            raise ValueError("hourly costs must be finite and nonnegative")
        return self


@dataclass(frozen=True)
class Measurement:
    batch_size: int
    batch_count: int
    processed_tickets: int
    repetitions: int
    durations_seconds: tuple[float, ...]
    total_measured_seconds: float
    p50_batch_seconds: float
    p95_batch_seconds: float
    throughput_tickets_per_second: float
    amortized_seconds_per_ticket: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "processed_tickets": self.processed_tickets,
            "repetitions": self.repetitions,
            "durations_seconds": list(self.durations_seconds),
            "total_measured_seconds": self.total_measured_seconds,
            "p50_batch_seconds": self.p50_batch_seconds,
            "p95_batch_seconds": self.p95_batch_seconds,
            "throughput_tickets_per_second": self.throughput_tickets_per_second,
            "amortized_seconds_per_ticket": self.amortized_seconds_per_ticket,
        }


def percentile(values: Sequence[float], percentage: float) -> float:
    """Linear percentile compatible with NumPy's default, without a dependency."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("durations must be finite and nonnegative")
    position = (len(ordered) - 1) * percentage / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_batches(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    if not items:
        raise ValueError("benchmark input must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


def measure_batches(
    items: Sequence[Any],
    batch_size: int,
    infer: Callable[[Sequence[Any]], Any],
    *,
    warmup_iterations: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> Measurement:
    """Measure complete batches; initialization and report writing stay outside."""
    if warmup_iterations < 0 or repetitions < 1:
        raise ValueError("invalid warm-up or repetition count")
    batches = make_batches(items, batch_size)
    sync = synchronize or (lambda: None)
    for index in range(warmup_iterations):
        infer(batches[index % len(batches)])
        sync()
    durations: list[float] = []
    processed = 0
    for _ in range(repetitions):
        for batch in batches:
            sync()
            start = clock()
            infer(batch)
            sync()
            elapsed = clock() - start
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("clock produced an invalid duration")
            durations.append(elapsed)
            processed += len(batch)
    total = sum(durations)
    if total <= 0:
        raise ValueError("measured duration must be positive")
    return Measurement(
        batch_size=batch_size,
        batch_count=len(durations),
        processed_tickets=processed,
        repetitions=repetitions,
        durations_seconds=tuple(durations),
        total_measured_seconds=total,
        p50_batch_seconds=percentile(durations, 50),
        p95_batch_seconds=percentile(durations, 95),
        throughput_tickets_per_second=processed / total,
        amortized_seconds_per_ticket=total / processed,
    )


def deterministic_sample(records: Sequence[Any], count: int, seed: int) -> list[Any]:
    if not records or count < 1:
        raise ValueError("sampling requires records and a positive count")
    if count >= len(records):
        return list(records)
    indices = sorted(random.Random(seed).sample(range(len(records)), count))
    return [records[index] for index in indices]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required benchmark artifact is missing: {path}") from error


def resolve_artifact_path(reference: str | Path, run_root: Path) -> Path:
    path = Path(reference)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_root / path, run_root.parent.parent.parent / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Persisted policy references are repository-relative. Return the anchored
    # candidate when none exists so readiness errors remain actionable even if
    # the command was launched from outside the checkout.
    return candidates[-1]


def available_devices() -> dict[str, bool]:
    try:
        import torch
    except ImportError:
        return {"cpu": True, "cuda": False, "mps": False}
    return {
        "cpu": True,
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
    }


def validate_artifacts(config: BenchmarkConfig) -> dict[str, Any]:
    """Validate frozen data, policies, selected artifacts, and requested devices."""
    root = config.run_root
    selected = verify_phase29_policy(root)
    splits = load_frozen_splits(root / "splits")
    metadata = load_feature_metadata(root / "tfidf" / "feature_metadata.json")
    if (metadata.dataset_version, metadata.split_version) != (
        selected["dataset_version"], selected["split_version"]
    ):
        raise ValueError("TF-IDF, selected-model, and split identities differ")
    vectorizer_path = root / "tfidf" / "vectorizer.joblib"
    load_vectorizer(vectorizer_path)
    missing: list[str] = []
    resolved_models: dict[str, str] = {}
    for model in config.models:
        reference = selected["models"][model][
            "checkpoint" if model == "transformer" else "model_artifact"
        ]
        path = resolve_artifact_path(reference, root)
        if not path.exists():
            missing.append(f"{model}: {reference}")
        else:
            resolved_models[model] = str(path)
    if missing:
        raise FileNotFoundError(
            "selected Phase 2.10 inference artifacts are missing; restore these exact "
            "artifacts (do not substitute baseline checkpoints):\n- " + "\n- ".join(missing)
        )
    for model in config.models:
        artifact = Path(resolved_models[model])
        if model == "logistic_regression":
            estimator = load_logistic_regression(artifact)
            if list(estimator.classes_) != selected["label_order"]:
                raise ValueError("selected Logistic Regression label mapping mismatch")
        elif model == "linear_svm":
            estimator = load_linear_svm(artifact)
            if list(estimator.classes_) != selected["label_order"]:
                raise ValueError("selected Linear SVM label mapping mismatch")
        else:
            model_config = _read_json(artifact / "config.json")
            tokenizer_config = _read_json(artifact / "tokenizer_config.json")
            id2label = {str(index): label for index, label in enumerate(selected["label_order"])}
            if model_config.get("id2label") != id2label:
                raise ValueError("selected Transformer checkpoint label mapping mismatch")
            if not any((artifact / name).is_file() for name in
                       ("tokenizer.json", "vocab.txt", "tokenizer.model")):
                raise FileNotFoundError("selected Transformer checkpoint has no tokenizer vocabulary")
            expected_name = selected["models"]["transformer"]["model_name"]
            recorded_name = tokenizer_config.get("name_or_path", expected_name)
            if recorded_name not in (expected_name, str(artifact), str(artifact.resolve())):
                raise ValueError("selected Transformer tokenizer identity mismatch")
    devices = available_devices()
    requested = list(config.transformer_devices)
    if "transformer" in config.models:
        unavailable = [name for name in requested if not devices[name]]
        if unavailable:
            raise RuntimeError("explicitly requested device(s) unavailable: " + ", ".join(unavailable))
    # Validate policies without applying confidence processing to label timing.
    calibration = _read_json(root / "threshold_calibration" / "final_calibration_policy.json")
    thresholds = _read_json(root / "threshold_calibration" / "final_threshold_policy.json")
    if calibration.get("dataset_version") != selected["dataset_version"]:
        raise ValueError("calibration policy dataset identity mismatch")
    if thresholds.get("split_version") != selected["split_version"]:
        raise ValueError("threshold policy split identity mismatch")
    return {
        "dataset_version": selected["dataset_version"],
        "split_version": selected["split_version"],
        "feature_version": selected["feature_version"],
        "transformer_dataset_version": selected["transformer_dataset_version"],
        "label_order": selected["label_order"],
        "selected_models": selected["models"],
        "resolved_model_artifacts": resolved_models,
        "vectorizer": str(vectorizer_path.resolve()),
        "available_devices": devices,
        "threshold_status": thresholds.get("operational_threshold_status"),
        "calibration_methods": {
            name: details.get("calibration_method")
            for name, details in calibration.get("models", {}).items()
        },
        "split_count": len(getattr(splits, config.split)),
    }


def _sync_for(device: str) -> Callable[[], None]:
    if device == "cpu":
        return lambda: None
    import torch
    if device == "cuda":
        return torch.cuda.synchronize
    synchronize = getattr(torch.mps, "synchronize", None)
    if synchronize is None:
        raise RuntimeError("installed PyTorch does not provide MPS synchronization")
    return synchronize


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _device_memory(device: str) -> dict[str, int | None]:
    if device == "cpu":
        return {"process_peak_rss_bytes": _rss_bytes(), "accelerator_peak_bytes": None}
    import torch
    if device == "cuda":
        return {
            "process_peak_rss_bytes": _rss_bytes(),
            "accelerator_peak_bytes": int(torch.cuda.max_memory_allocated()),
        }
    current = getattr(torch.mps, "current_allocated_memory", None)
    return {
        "process_peak_rss_bytes": _rss_bytes(),
        "accelerator_peak_bytes": int(current()) if current else None,
    }


def artifact_sizes(paths: Iterable[str | Path]) -> dict[str, Any]:
    unique: dict[str, int] = {}
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"cannot size missing artifact: {path}")
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for file in files:
            unique[str(file.resolve())] = file.stat().st_size
    return {"files": unique, "total_bytes": sum(unique.values())}


def calculate_costs(throughput: float, hourly_rate: float | None) -> dict[str, float] | None:
    if hourly_rate is None:
        return None
    if throughput <= 0 or not math.isfinite(throughput):
        raise ValueError("throughput must be positive and finite")
    if hourly_rate < 0 or not math.isfinite(hourly_rate):
        raise ValueError("hourly rate must be finite and nonnegative")
    per_ticket = hourly_rate / (throughput * 3600)
    return {
        "hourly_rate": hourly_rate,
        "utilization_assumption": 1.0,
        "cost_per_ticket": per_ticket,
        "cost_per_1000_tickets": per_ticket * 1_000,
        "cost_per_1000000_tickets": per_ticket * 1_000_000,
    }


def _benchmark_classical(model_name: str, config: BenchmarkConfig, readiness: dict, texts: list[str]):
    model_path = Path(readiness["resolved_model_artifacts"][model_name])
    model = load_logistic_regression(model_path) if model_name == "logistic_regression" else load_linear_svm(model_path)
    vectorizer = load_vectorizer(readiness["vectorizer"])
    if list(model.classes_) != readiness["label_order"]:
        raise ValueError(f"{model_name} label mapping differs from frozen label order")

    def end_to_end(batch):
        predictions = model.predict(vectorizer.transform(batch))
        if len(predictions) != len(batch):
            raise ValueError("classical prediction shape mismatch")
        return predictions

    results = []
    rss_before = _rss_bytes()
    for size in config.batch_sizes:
        measurement = measure_batches(texts, size, end_to_end,
            warmup_iterations=config.warmup_iterations, repetitions=config.repetitions)
        row = measurement.as_dict()
        row["cost"] = calculate_costs(row["throughput_tickets_per_second"], config.hourly_costs.get("cpu"))
        results.append(row)
    return {
        "model": model_name, "device": "cpu", "timing_scope": "raw text -> TF-IDF -> predict -> label array",
        "measurements": results,
        "memory": {**_device_memory("cpu"), "peak_rss_delta_lower_bound_bytes": max(0, _rss_bytes() - rss_before)},
        "artifact_size": artifact_sizes((model_path, readiness["vectorizer"])),
        "weights_size": artifact_sizes((model_path,)),
    }


def _benchmark_transformer(config: BenchmarkConfig, readiness: dict, texts: list[str], device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ticket_classification.transformer_training import TransformerTrainingConfig, validate_model_configuration

    checkpoint = Path(readiness["resolved_model_artifacts"]["transformer"])
    training_config = TransformerTrainingConfig.model_validate(
        _read_json(config.run_root / "transformer_training" / "training_config.json")
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    validate_model_configuration(model, training_config)
    if tokenizer.pad_token_id is None:
        raise ValueError("selected tokenizer has no padding token")
    target = torch.device(device)
    model.to(target).eval()
    sync = _sync_for(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    def end_to_end(batch):
        encoded = tokenizer(list(batch), padding=True, truncation=True,
            max_length=training_config.max_length, return_tensors="pt")
        encoded = {name: tensor.to(target) for name, tensor in encoded.items()}
        with torch.inference_mode():
            predictions = model(**encoded).logits.argmax(dim=-1)
        if predictions.shape != (len(batch),):
            raise ValueError("Transformer prediction shape mismatch")
        return predictions

    results = []
    rss_before = _rss_bytes()
    for size in config.batch_sizes:
        measurement = measure_batches(texts, size, end_to_end,
            warmup_iterations=config.warmup_iterations, repetitions=config.repetitions,
            synchronize=sync)
        row = measurement.as_dict()
        row["cost"] = calculate_costs(row["throughput_tickets_per_second"], config.hourly_costs.get(device))
        results.append(row)
    weight_files = tuple(checkpoint.glob("*.safetensors")) or tuple(checkpoint.glob("pytorch_model*.bin"))
    inference_files = tuple(
        path for path in checkpoint.iterdir()
        if path.is_file() and path.name != "training_state.pt"
    )
    return {
        "model": "transformer", "device": device,
        "timing_scope": "raw text -> tokenization/dynamic padding -> device transfer -> model -> argmax",
        "measurements": results,
        "memory": {**_device_memory(device), "peak_rss_delta_lower_bound_bytes": max(0, _rss_bytes() - rss_before)},
        "artifact_size": artifact_sizes(inference_files),
        "weights_size": artifact_sizes(weight_files),
        "max_length": training_config.max_length,
    }


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "sklearn", "torch", "transformers", "joblib"):
        try:
            module = __import__(module_name)
            versions[module_name] = module.__version__
        except ImportError:
            versions[module_name] = "unavailable"
    return {
        "python": platform.python_version(), "libraries": versions,
        "operating_system": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "available_devices": available_devices(),
    }


def run_benchmarks(config: BenchmarkConfig) -> dict[str, Any]:
    readiness = validate_artifacts(config)
    splits = load_frozen_splits(config.run_root / "splits")
    records = deterministic_sample(getattr(splits, config.split), config.sample_count, config.random_seed)
    texts = [build_classification_text(record) for record in records]
    devices = list(config.transformer_devices)
    available = readiness["available_devices"]
    if config.include_available_accelerator and "transformer" in config.models:
        devices.extend(name for name in ("cuda", "mps") if available[name] and name not in devices)
    results = []
    for model_name in config.models:
        if model_name == "transformer":
            results.extend(_benchmark_transformer(config, readiness, texts, device) for device in devices)
        else:
            results.append(_benchmark_classical(model_name, config, readiness, texts))
    lengths = [len(text) for text in texts]
    return {
        "schema_version": "phase_2_11_v1", "executed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config.model_dump(mode="json"), "artifact_validation": readiness,
        "environment": runtime_environment(),
        "sample": {"dataset_version": readiness["dataset_version"], "split": config.split,
            "sampling_policy": "seeded sample without replacement; source order restored", "random_seed": config.random_seed,
            "count": len(records), "record_ids": [r.record_id for r in records],
            "input_character_lengths": {"minimum": min(lengths), "maximum": max(lengths),
                "p50": percentile(lengths, 50), "p95": percentile(lengths, 95)}},
        "results": results,
        "limitations": [
            "Process peak RSS is process-wide, not model-only, and the delta is a lower bound.",
            "MPS exposes current allocated memory here, not a reliable peak metric.",
            "Hosted API pricing is not applicable; all selected classifiers are local models.",
            "Calibration affects confidence values, not argmax labels; pending thresholds are not applied.",
        ],
    }


def write_results(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output / "benchmark_config.json", json.dumps(
        config.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False) + "\n")
    _atomic_write_text(output / "benchmark_results.json", json.dumps(
        results, sort_keys=True, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema_version": results["schema_version"], "executed_at": results["executed_at"],
        "dataset_version": results["sample"]["dataset_version"],
        "split": results["sample"]["split"], "sample_count": results["sample"]["count"],
        "models": config.models, "artifacts": ["benchmark_config.json", "benchmark_results.json", "inference_benchmark.md"],
    }
    _atomic_write_text(output / "benchmark_manifest.json", json.dumps(
        manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _atomic_write_text(output / "inference_benchmark.md", render_report(results))


def render_report(data: dict[str, Any]) -> str:
    sample, environment = data["sample"], data["environment"]
    lines = ["# Phase 2.11 Inference Benchmark", "", "## Executive summary", "",
        "This report contains measured local inference results only. Compare devices separately and choose operating settings from the observed rows, not a hardcoded optimum.", "",
        "## Benchmark objective", "", "Compare steady-state end-to-end latency, throughput, artifact footprint, memory observations, and optional infrastructure cost for the three frozen selected classifiers.", "",
        "## Selected model identities", "", "```json", json.dumps(data["artifact_validation"]["selected_models"], indent=2, sort_keys=True), "```", "",
        "## Dataset and sample information", "", f"- Dataset: `{sample['dataset_version']}`", f"- Split: `{sample['split']}`", f"- Sample count: {sample['count']}", f"- Sampling: {sample['sampling_policy']} (seed {sample['random_seed']})", f"- Input character lengths: `{json.dumps(sample['input_character_lengths'], sort_keys=True)}`", "",
        "## Hardware and runtime environment", "", "```json", json.dumps(environment, indent=2, sort_keys=True), "```", "",
        "## Benchmark methodology", "", "Model initialization and artifact loading are excluded. Primary timing includes text vectorization/tokenization, transfers, model execution, and prediction extraction. Accelerator work is synchronized before and after each measurement. Throughput uses actual processed ticket counts divided by total measured duration.", ""]
    for result in data["results"]:
        title = f"{result['model']} on {result['device']}"
        lines.extend([f"## {title}", "", f"Timing scope: {result['timing_scope']}", "", "| Batch size | P50 batch (ms) | P95 batch (ms) | Tickets/s | Amortized ms/ticket | Measured tickets |", "|---:|---:|---:|---:|---:|---:|"])
        for row in result["measurements"]:
            lines.append(f"| {row['batch_size']} | {row['p50_batch_seconds']*1000:.3f} | {row['p95_batch_seconds']*1000:.3f} | {row['throughput_tickets_per_second']:.3f} | {row['amortized_seconds_per_ticket']*1000:.3f} | {row['processed_tickets']} |")
        lines.extend(["", "### Serialized model sizes", "", f"- Model weights: {result['weights_size']['total_bytes']} bytes", f"- Complete inference artifact: {result['artifact_size']['total_bytes']} bytes", "", "### Memory measurements", "", "```json", json.dumps(result["memory"], indent=2, sort_keys=True), "```", "", "### Operational cost assumptions", ""])
        costs = [(row["batch_size"], row["cost"]) for row in result["measurements"] if row["cost"] is not None]
        if not costs:
            lines.append("No hourly infrastructure cost was supplied; calculated costs are N/A. Hosted API pricing is not applicable.")
        else:
            lines.extend(["| Batch size | Hourly rate | Cost/ticket | Cost/1K | Cost/1M |", "|---:|---:|---:|---:|---:|"])
            for batch, cost in costs:
                lines.append(f"| {batch} | {cost['hourly_rate']:.6g} | {cost['cost_per_ticket']:.9g} | {cost['cost_per_1000_tickets']:.9g} | {cost['cost_per_1000000_tickets']:.9g} |")
        lines.append("")
    lines.extend(["## CPU single-example latency", "", "Rows with batch size 1 above are the CPU single-example results.", "", "## CPU batch inference latency", "", "CPU rows with batch sizes greater than 1 show batch-size impact, P50/P95 latency, and throughput.", "", "## Transformer accelerator results", "", "CUDA/MPS sections are present only for devices actually benchmarked; unsupported accelerators are not fabricated.", "", "## P50, P95, throughput, and batch-size impact", "", "Every result table reports P50, P95, throughput from actual record counts, and amortized per-ticket time.", "", "## Operational requirements and complexity", "", "Classical inference requires CPU, fitted TF-IDF assets, and its selected estimator. Transformer inference requires PyTorch, Transformers, the complete local checkpoint/tokenizer, more runtime memory, and optional CUDA/MPS device management. No hosted API is part of this system.", "", "## Observed performance trade-offs", "", "The tables above are the authoritative observations. Batch scaling, tail latency, throughput, footprint, memory, device requirements, and supplied cost rates should be evaluated together; larger batches are not assumed to be better.", "", "## Limitations and unavailable measurements", ""])
    lines.extend(f"- {item}" for item in data["limitations"])
    lines.extend(["", "## Reproduction commands", "", "See the repository Phase 2.11 handoff instructions and run `python -m ticket_classification.run_benchmark --help` for the exact CLI.", ""])
    return "\n".join(lines)


def regenerate_report(results_path: Path, output_path: Path | None = None) -> Path:
    data = _read_json(results_path)
    target = output_path or results_path.with_name("inference_benchmark.md")
    _atomic_write_text(target, render_report(data))
    return target

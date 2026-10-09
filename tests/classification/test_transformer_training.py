"""Focused Phase 2.8 tests with a tiny local PyTorch classifier."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")

from ticket_classification.transformer_training import (
    DEFAULT_LABEL_ORDER,
    BestCheckpoint,
    DynamicPaddingCollator,
    TrainingHistoryEntry,
    TransformerTrainingConfig,
    build_data_loader,
    compute_transformer_metrics,
    evaluate_model,
    load_test_data,
    load_training_data,
    run_final_test_evaluation,
    run_training,
    select_best_checkpoint,
    validate_model_configuration,
    validate_only,
)


DATASET_DIR = Path("artifacts/classification/run/transformer")


class TinyTokenizer:
    pad_token_id = 0

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text('{"pad_token_id":0}\n')


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(31_000, 8, padding_idx=0)
        self.classifier = torch.nn.Linear(8, 4)
        self.config = SimpleNamespace(
            num_labels=4,
            label2id={label: index for index, label in enumerate(DEFAULT_LABEL_ORDER)},
            id2label={index: label for index, label in enumerate(DEFAULT_LABEL_ORDER)},
            problem_type="single_label_classification",
        )

    def forward(self, input_ids, attention_mask, labels=None):
        vectors = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (vectors * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.classifier(pooled)
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, directory, safe_serialization=True):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "tiny_model.pt")
        (path / "config.json").write_text('{"num_labels":4}\n')


def test_training_configuration_is_frozen_and_unweighted() -> None:
    config = TransformerTrainingConfig()
    assert config.epochs == 3
    assert config.train_batch_size == 16
    assert config.eval_batch_size == 32
    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.01
    assert config.best_model_metric == "macro_f1"
    assert config.class_weight is None
    assert config.early_stopping_enabled is False
    with pytest.raises(ValidationError):
        TransformerTrainingConfig(class_weight="balanced")
    with pytest.raises(ValidationError):
        TransformerTrainingConfig(label_order=("Incident", "Change", "Problem", "Request"))


def test_model_configuration_has_exact_labels() -> None:
    model = TinyClassifier()
    validate_model_configuration(model, TransformerTrainingConfig())
    assert model.config.num_labels == 4
    assert model.config.label2id == {"Change": 0, "Incident": 1, "Problem": 2, "Request": 3}


def test_real_phase27_train_validation_and_test_load_independently() -> None:
    config = TransformerTrainingConfig()
    training = load_training_data(DATASET_DIR, config)
    assert (len(training.train), len(training.validation)) == (20_010, 4_288)
    assert training.labels.label_order == DEFAULT_LABEL_ORDER
    test, second = load_test_data(DATASET_DIR, config)
    assert len(test) == 4_289
    assert second.manifest.transformer_dataset_version == config.transformer_dataset_version


def test_dynamic_padding_shapes_and_values() -> None:
    rows = [
        {"record_id": "a", "input_ids": [101, 10, 102], "attention_mask": [1, 1, 1], "labels": 0},
        {"record_id": "b", "input_ids": [101, 102], "attention_mask": [1, 1], "labels": 3},
    ]
    batch = DynamicPaddingCollator(0)(rows)
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["attention_mask"].shape) == (2, 3)
    assert batch["input_ids"][1].tolist() == [101, 102, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"].tolist() == [0, 3]


def test_forward_logits_loss_metrics_and_predictions() -> None:
    training = load_training_data(DATASET_DIR, TransformerTrainingConfig())
    loader = build_data_loader(
        training.validation[:5], 5, DynamicPaddingCollator(0), shuffle=False, seed=42
    )
    result = evaluate_model(TinyClassifier(), loader, torch.device("cpu"), DEFAULT_LABEL_ORDER)
    assert len(result.predictions) == 5
    assert result.loss > 0
    assert all(len(row["logits"]) == 4 for row in result.predictions)
    assert all(len(row["probabilities"]) == 4 for row in result.predictions)
    metrics, confusion = compute_transformer_metrics([0, 1, 2, 3], [0, 1, 3, 3], DEFAULT_LABEL_ORDER)
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((1 + 1 + 0 + 2 / 3) / 4)
    assert confusion["row_labels"] == list(DEFAULT_LABEL_ORDER)


def test_best_checkpoint_uses_macro_f1() -> None:
    first = BestCheckpoint(path="first", epoch=1, step=10, validation_macro_f1=0.7)
    worse = BestCheckpoint(path="worse", epoch=2, step=20, validation_macro_f1=0.6)
    better = BestCheckpoint(path="better", epoch=3, step=30, validation_macro_f1=0.8)
    assert select_best_checkpoint(None, first) == first
    assert select_best_checkpoint(first, worse) == first
    assert select_best_checkpoint(first, better) == better


def test_training_history_serializes_required_fields() -> None:
    row = TrainingHistoryEntry(
        epoch=1, step=4, train_loss=1.2, validation_loss=1.1, learning_rate=1e-5,
        validation_accuracy=0.5, validation_macro_precision=0.4,
        validation_macro_recall=0.3, validation_macro_f1=0.35,
        validation_weighted_f1=0.45,
    )
    assert json.loads(row.model_dump_json())["validation_macro_f1"] == 0.35


def test_validate_only_does_not_update_parameters(tmp_path, monkeypatch) -> None:
    model = TinyClassifier()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(
        "ticket_classification.transformer_training.load_model_and_tokenizer",
        lambda config: (model, TinyTokenizer()),
    )
    result = validate_only(DATASET_DIR, tmp_path)
    assert result["optimizer_steps"] == 0
    assert result["logits_shape"] == [2, 4]
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_smoke_training_is_limited_saves_and_reloads(tmp_path, monkeypatch) -> None:
    initial = TinyClassifier()

    def reload_model(directory):
        model = TinyClassifier()
        model.load_state_dict(torch.load(Path(directory) / "tiny_model.pt", map_location="cpu"))
        return model

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained", reload_model
    )
    manifest = run_training(
        DATASET_DIR,
        tmp_path,
        smoke_test=True,
        model_and_tokenizer=(initial, TinyTokenizer()),
    )
    assert manifest["smoke_test"] is True
    assert manifest["train_count"] == 32
    assert manifest["validation_count"] == 16
    assert manifest["test_set_loaded"] is False
    assert (tmp_path / "best_model" / "tiny_model.pt").exists()
    assert (tmp_path / "validation_predictions.jsonl").read_text().count("\n") == 16
    history = json.loads((tmp_path / "training_history.json").read_text())
    assert len(history) == 1 and history[0]["step"] == 4


def test_final_test_evaluation_is_inference_only_and_persists_outputs(tmp_path, monkeypatch) -> None:
    config = TransformerTrainingConfig(eval_batch_size=2)
    test, training = load_test_data(DATASET_DIR, config)
    best_model = tmp_path / "best_model"
    TinyClassifier().save_pretrained(best_model)
    TinyTokenizer().save_pretrained(best_model)
    (tmp_path / "training_manifest.json").write_text(
        json.dumps(
            {
                "smoke_test": False,
                "test_set_loaded": False,
                "dataset_version": config.dataset_version,
                "split_version": config.split_version,
                "transformer_dataset_version": config.transformer_dataset_version,
                "train_count": 20_010,
                "validation_count": 4_288,
                "configuration": config.model_dump(mode="json"),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "ticket_classification.transformer_training.load_test_data",
        lambda dataset_dir, settings: (test[:5], training),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        lambda directory: TinyClassifier(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda directory: TinyTokenizer(),
    )
    report = run_final_test_evaluation(DATASET_DIR, tmp_path, config)
    assert report["inference_only"] is True
    assert report["checkpoint_selection_changed"] is False
    assert report["test_count"] == 5
    assert (tmp_path / "test_predictions.jsonl").read_text().count("\n") == 5
    assert (tmp_path / "test_metrics.json").exists()
    assert (tmp_path / "test_confusion_matrix.json").exists()

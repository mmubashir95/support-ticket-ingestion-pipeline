import os

import numpy as np
import pytest

from ticket_pipeline.semantic_deduplication import (
    DEFAULT_EMBEDDING_MODEL,
    build_semantic_text,
    detect_semantic_duplicate_candidates,
    find_semantic_duplicate_candidates,
    generate_embeddings,
)


class StubEmbeddingModel:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text
        self.calls = 0
        self.encoded_texts: list[str] = []
        self.batch_size: int | None = None

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.calls += 1
        self.encoded_texts = sentences
        self.batch_size = batch_size
        assert normalize_embeddings is True
        assert show_progress_bar is False
        assert convert_to_numpy is True
        return np.asarray(
            [self.vectors_by_text[text] for text in sentences],
            dtype=np.float32,
        )


def test_semantic_text_uses_subject_and_body() -> None:
    assert build_semantic_text("Login issue", "Cannot login") == (
        "Login issue\nCannot login"
    )
    assert build_semantic_text("Different subject", "Cannot login") != (
        build_semantic_text("Login issue", "Cannot login")
    )
    assert build_semantic_text("Login issue", "Different body") != (
        build_semantic_text("Login issue", "Cannot login")
    )
    assert build_semantic_text(None, "Cannot login") == "\nCannot login"


def test_embedding_generation_is_batched_normalized_and_one_row_per_text() -> None:
    texts = [
        "Payment failed 😡",
        "José cannot login",
        "München account issue",
        "Contact <EMAIL> Call <PHONE> Server <IP_ADDRESS>",
    ]
    model = StubEmbeddingModel(
        {
            texts[0]: [3.0, 4.0],
            texts[1]: [1.0, 0.0],
            texts[2]: [0.0, 2.0],
            texts[3]: [2.0, 2.0],
        }
    )

    embeddings = generate_embeddings(texts, batch_size=2, model=model)

    assert model.calls == 1
    assert model.encoded_texts == texts
    assert model.batch_size == 2
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_zero_tickets_returns_empty_results_without_loading_model() -> None:
    assert detect_semantic_duplicate_candidates([]) == []


def test_one_ticket_has_no_semantic_candidate() -> None:
    text = build_semantic_text("Login", "Cannot login")
    model = StubEmbeddingModel({text: [1.0, 0.0]})

    results = detect_semantic_duplicate_candidates(
        [("Login", "Cannot login")], model=model
    )

    assert results[0]["is_semantic_duplicate_candidate"] is False
    assert results[0]["candidate_duplicate_of"] is None
    assert results[0]["semantic_similarity"] is None


def test_clear_paraphrase_is_candidate_but_different_issue_is_not() -> None:
    tickets = [
        ("Login issue", "I cannot login to my account"),
        ("Sign-in issue", "I'm unable to sign in to my account"),
        ("Password issue", "I cannot reset my password"),
    ]
    texts = [build_semantic_text(*ticket) for ticket in tickets]
    model = StubEmbeddingModel(
        {
            texts[0]: [1.0, 0.0, 0.0],
            texts[1]: [0.98, 0.2, 0.0],
            texts[2]: [0.0, 1.0, 0.0],
        }
    )

    results = detect_semantic_duplicate_candidates(
        tickets, threshold=0.9, top_k=2, model=model
    )

    assert results[1]["candidate_duplicate_of"] == 0
    assert results[1]["semantic_similarity"] is not None
    assert results[1]["semantic_similarity"] > 0.9
    assert results[2]["is_semantic_duplicate_candidate"] is False


def test_payment_paraphrase_ranks_above_refund_request() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],  # My payment keeps failing
            [0.95, 0.2],  # I cannot complete the payment
            [0.0, 1.0],  # I want a refund
        ]
    )

    results = find_semantic_duplicate_candidates(
        embeddings, threshold=0.9, top_k=2
    )

    assert results[1]["candidate_duplicate_of"] == 0
    assert results[2]["is_semantic_duplicate_candidate"] is False


def test_self_matches_are_excluded_and_direction_is_later_to_earlier() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.1]])

    results = find_semantic_duplicate_candidates(
        embeddings, threshold=0.9, top_k=1
    )

    assert results[0]["candidate_duplicate_of"] is None
    assert results[1]["candidate_duplicate_of"] == 0
    assert all(
        result["record_index"] != result["candidate_duplicate_of"]
        for result in results
    )


def test_highest_similarity_earlier_neighbor_is_selected() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.95, 0.1],
        ]
    )

    results = find_semantic_duplicate_candidates(
        embeddings, threshold=0.8, top_k=2
    )

    assert results[2]["candidate_duplicate_of"] == 0


def test_threshold_is_configurable_and_score_is_exposed() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.4358899]])

    lower = find_semantic_duplicate_candidates(
        embeddings, threshold=0.85, top_k=1
    )
    higher = find_semantic_duplicate_candidates(
        embeddings, threshold=0.95, top_k=1
    )

    assert lower[1]["is_semantic_duplicate_candidate"] is True
    assert higher[1]["is_semantic_duplicate_candidate"] is False
    assert lower[1]["semantic_similarity"] == pytest.approx(0.9, abs=1e-5)
    assert lower[1]["threshold"] == 0.85


@pytest.mark.parametrize("threshold", [-1.01, 1.01])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        find_semantic_duplicate_candidates(
            np.asarray([[1.0, 0.0]]), threshold=threshold
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        find_semantic_duplicate_candidates(
            np.asarray([[1.0, 0.0]]), top_k=top_k  # type: ignore[arg-type]
        )


def test_top_k_larger_than_dataset_is_safe() -> None:
    results = find_semantic_duplicate_candidates(
        np.asarray([[1.0, 0.0], [0.99, 0.1]]),
        threshold=0.9,
        top_k=50,
    )

    assert len(results) == 2
    assert results[1]["candidate_duplicate_of"] == 0


def test_exact_duplicates_are_skipped_from_embedding_and_semantic_results() -> None:
    tickets = [
        ("Login", "Cannot login"),
        ("Login", "Cannot login"),
        ("Refund", "Need a refund"),
    ]
    canonical_texts = [
        build_semantic_text(*tickets[0]),
        build_semantic_text(*tickets[2]),
    ]
    model = StubEmbeddingModel(
        {
            canonical_texts[0]: [1.0, 0.0],
            canonical_texts[1]: [0.0, 1.0],
        }
    )

    results = detect_semantic_duplicate_candidates(
        tickets,
        exact_duplicate_of=[None, 0, None],
        threshold=0.9,
        model=model,
    )

    assert model.encoded_texts == canonical_texts
    assert results[1]["skipped_exact_duplicate"] is True
    assert results[1]["is_semantic_duplicate_candidate"] is False
    assert results[1]["semantic_similarity"] is None


def test_record_indices_preserve_original_direction_after_exact_skip() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.1]])

    results = find_semantic_duplicate_candidates(
        embeddings,
        record_indices=[0, 2],
        threshold=0.9,
        top_k=1,
    )

    assert [result["record_index"] for result in results] == [0, 2]
    assert results[1]["candidate_duplicate_of"] == 0


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_SEMANTIC_MODEL_TEST") != "1",
    reason="set RUN_REAL_SEMANTIC_MODEL_TEST=1 to allow model loading/download",
)
def test_real_sentence_transformer_smoke() -> None:
    texts = [
        "I cannot login to my account",
        "I'm unable to sign in to my account",
        "I want a refund for my order",
    ]

    embeddings = generate_embeddings(texts)
    similarities = embeddings @ embeddings.T

    assert embeddings.shape[0] == 3
    assert similarities[0, 1] > similarities[0, 2]
    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

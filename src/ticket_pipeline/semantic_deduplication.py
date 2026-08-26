"""Embedding-based semantic duplicate candidate detection.

Inputs must already be normalized, PII-masked, and exact-deduplicated. The
default threshold is a provisional operational value, not a calibrated claim.
"""

from collections.abc import Sequence
from typing import Protocol, TypedDict

import numpy as np
from sklearn.neighbors import NearestNeighbors


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD = 0.85
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32


class EmbeddingModel(Protocol):
    """Minimal sentence-transformer-compatible encoding interface."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


class SemanticDuplicateResult(TypedDict):
    """Traceable semantic-candidate metadata for one input record."""

    record_index: int
    is_semantic_duplicate_candidate: bool
    candidate_duplicate_of: int | None
    nearest_earlier_record: int | None
    semantic_similarity: float | None
    threshold: float
    skipped_exact_duplicate: bool


def build_semantic_text(subject: str | None, body: str) -> str:
    """Join a processed subject and body for sentence embedding."""

    canonical_subject = subject if subject is not None else ""
    return f"{canonical_subject}\n{body}"


def _validate_configuration(threshold: float, top_k: int) -> None:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.floating))
        or not -1.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be between -1.0 and 1.0")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _normalize_embedding_matrix(embeddings: object) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")

    if matrix.shape[0] == 0:
        return matrix

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must have non-zero magnitude")
    return matrix / norms


def generate_embeddings(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: EmbeddingModel | None = None,
) -> np.ndarray:
    """Generate one normalized embedding per text in batched CPU inference.

    Supplying ``model`` supports deterministic offline tests. Otherwise the
    configured sentence-transformer is loaded once for this operation.
    """

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embedding_model = model
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for real embedding inference"
            ) from error
        embedding_model = SentenceTransformer(model_name, device="cpu")

    embeddings = embedding_model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    matrix = _normalize_embedding_matrix(embeddings)
    if matrix.shape[0] != len(texts):
        raise ValueError("embedding model must return one row per input text")
    return matrix


def _empty_result(
    record_index: int,
    threshold: float,
    *,
    skipped_exact_duplicate: bool = False,
) -> SemanticDuplicateResult:
    return {
        "record_index": record_index,
        "is_semantic_duplicate_candidate": False,
        "candidate_duplicate_of": None,
        "nearest_earlier_record": None,
        "semantic_similarity": None,
        "threshold": threshold,
        "skipped_exact_duplicate": skipped_exact_duplicate,
    }


def find_semantic_duplicate_candidates(
    embeddings: object,
    *,
    record_indices: Sequence[int] | None = None,
    threshold: float = DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[SemanticDuplicateResult]:
    """Select each record's best retrieved earlier neighbor using cosine similarity."""

    _validate_configuration(threshold, top_k)
    matrix = _normalize_embedding_matrix(embeddings)
    record_count = matrix.shape[0]

    indices = (
        list(range(record_count))
        if record_indices is None
        else list(record_indices)
    )
    if len(indices) != record_count:
        raise ValueError("record_indices must match the embedding row count")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("record_indices must contain integers")
    if any(current <= previous for previous, current in zip(indices, indices[1:])):
        raise ValueError("record_indices must be strictly increasing")

    results = [_empty_result(index, threshold) for index in indices]
    if record_count < 2:
        return results

    neighbor_count = min(record_count, top_k + 1)
    neighbor_search = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
    )
    distances, neighbor_positions = neighbor_search.fit(matrix).kneighbors(matrix)

    for row_position, record_index in enumerate(indices):
        earlier_neighbors: list[tuple[float, int]] = []
        for distance, neighbor_position in zip(
            distances[row_position], neighbor_positions[row_position]
        ):
            if int(neighbor_position) == row_position:
                continue
            neighbor_index = indices[int(neighbor_position)]
            if neighbor_index >= record_index:
                continue
            similarity = float(np.clip(1.0 - distance, -1.0, 1.0))
            earlier_neighbors.append((similarity, neighbor_index))

        if not earlier_neighbors:
            continue

        similarity, nearest_index = max(
            earlier_neighbors,
            key=lambda match: (match[0], -match[1]),
        )
        is_candidate = similarity >= threshold
        results[row_position] = {
            "record_index": record_index,
            "is_semantic_duplicate_candidate": is_candidate,
            "candidate_duplicate_of": nearest_index if is_candidate else None,
            "nearest_earlier_record": nearest_index,
            "semantic_similarity": similarity,
            "threshold": threshold,
            "skipped_exact_duplicate": False,
        }

    return results


def detect_semantic_duplicate_candidates(
    tickets: Sequence[tuple[str | None, str]],
    *,
    exact_duplicate_of: Sequence[int | None] | None = None,
    threshold: float = DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model: EmbeddingModel | None = None,
) -> list[SemanticDuplicateResult]:
    """Embed exact-unique processed tickets and return aligned candidate results."""

    _validate_configuration(threshold, top_k)
    ticket_count = len(tickets)
    exact_links = (
        [None] * ticket_count
        if exact_duplicate_of is None
        else list(exact_duplicate_of)
    )
    if len(exact_links) != ticket_count:
        raise ValueError("exact_duplicate_of must match the ticket count")

    for record_index, duplicate_of in enumerate(exact_links):
        is_valid_link = (
            duplicate_of is None
            or (
                not isinstance(duplicate_of, bool)
                and isinstance(duplicate_of, int)
                and 0 <= duplicate_of < record_index
            )
        )
        if not is_valid_link:
            raise ValueError("exact duplicates must point to an earlier record index")

    canonical_indices = [
        index for index, duplicate_of in enumerate(exact_links) if duplicate_of is None
    ]
    semantic_texts = [
        build_semantic_text(*tickets[index]) for index in canonical_indices
    ]
    embeddings = generate_embeddings(
        semantic_texts,
        model_name=model_name,
        batch_size=batch_size,
        model=model,
    )
    canonical_results = find_semantic_duplicate_candidates(
        embeddings,
        record_indices=canonical_indices,
        threshold=threshold,
        top_k=top_k,
    )
    result_by_index = {
        result["record_index"]: result for result in canonical_results
    }

    return [
        result_by_index[index]
        if exact_links[index] is None
        else _empty_result(index, threshold, skipped_exact_duplicate=True)
        for index in range(ticket_count)
    ]

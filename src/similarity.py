"""TF-IDF vectorization and resume/JD similarity utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(documents: Sequence[str]) -> np.ndarray:
    """Convert documents into a dense TF-IDF matrix."""
    docs = [str(document).strip() for document in documents]
    if not docs:
        raise ValueError("documents cannot be empty")

    if not any(docs):
        return np.zeros((len(docs), 0), dtype=float)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)(?<!\w)[\w+#.-]+(?!\w)",
    )

    try:
        return vectorizer.fit_transform(docs).toarray()
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            return np.zeros((len(docs), 0), dtype=float)
        raise


def cosine_scores_against_target(
    matrix: np.ndarray,
    target_index: int = 0,
) -> list[float]:
    """Calculate cosine similarity of each row against a target row."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not 0 <= target_index < matrix.shape[0]:
        raise IndexError("target_index is outside the matrix")
    if matrix.shape[1] == 0:
        return [0.0] * matrix.shape[0]

    target = matrix[target_index].reshape(1, -1)
    return cosine_similarity(matrix, target).ravel().astype(float).tolist()


def rank_resumes(
    resume_names: Sequence[str],
    scores: Sequence[float],
) -> list[tuple[str, float]]:
    """Return resumes sorted by descending similarity score."""
    if len(resume_names) != len(scores):
        raise ValueError("resume_names and scores must have equal length")

    ranked = [
        (str(name), float(score))
        for name, score in zip(resume_names, scores, strict=True)
    ]
    return sorted(
        ranked,
        key=lambda item: (-item[1], item[0].casefold()),
    )

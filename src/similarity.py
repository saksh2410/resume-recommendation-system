"""Resume/job-description vectorization and similarity utilities."""

from collections.abc import Sequence

import numpy as np


def build_tfidf_matrix(documents: Sequence[str]) -> np.ndarray:
    """Convert documents into TF-IDF vectors."""
    raise NotImplementedError


def cosine_scores_against_target(
    matrix: np.ndarray,
    target_index: int = 0,
) -> list[float]:
    """Calculate cosine similarity of each row against a target row."""
    raise NotImplementedError


def rank_resumes(
    resume_names: Sequence[str],
    scores: Sequence[float],
) -> list[tuple[str, float]]:
    """Return resumes sorted by descending similarity score."""
    raise NotImplementedError

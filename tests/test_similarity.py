"""Tests for TF-IDF and cosine-similarity utilities."""

import pytest

from src.similarity import (
    build_tfidf_matrix,
    cosine_scores_against_target,
    rank_resumes,
)


def test_similarity_ranks_closer_profile_higher() -> None:
    documents = [
        "Python machine learning Docker",
        "Python machine learning",
        "graphic design sales",
    ]

    matrix = build_tfidf_matrix(documents)
    scores = cosine_scores_against_target(matrix)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] > scores[2]


def test_empty_documents_return_zero_scores() -> None:
    matrix = build_tfidf_matrix(["", ""])
    scores = cosine_scores_against_target(matrix)

    assert matrix.shape == (2, 0)
    assert scores == [0.0, 0.0]


def test_rank_resumes_orders_by_score() -> None:
    ranked = rank_resumes(
        ["b.pdf", "a.pdf"],
        [0.2, 0.9],
    )

    assert ranked == [
        ("a.pdf", 0.9),
        ("b.pdf", 0.2),
    ]

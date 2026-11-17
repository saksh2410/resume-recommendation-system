"""Tests for domain skill extraction."""

from pathlib import Path

from src.skills import (
    extract_known_skills,
    load_skill_vocabulary,
)


def test_load_skill_vocabulary_deduplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skills.txt"
    path.write_text(
        "Python\nSQL\npython\n\nC++\n",
        encoding="utf-8",
    )

    assert load_skill_vocabulary(path) == [
        "Python",
        "SQL",
        "C++",
    ]


def test_extract_known_skills_handles_boundaries() -> None:
    vocabulary = [
        "Python",
        "R",
        "Go",
        "C++",
        "project management",
    ]
    text = (
        "Built Python and C++ services and supported "
        "project management. Growing teams should not "
        "count as Go or R."
    )

    skills = extract_known_skills(text, vocabulary)

    assert skills == [
        "Python",
        "C++",
        "project management",
    ]

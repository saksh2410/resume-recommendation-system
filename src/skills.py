"""Skill candidate extraction utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def load_skill_vocabulary(path: str | Path) -> list[str]:
    """Load a newline-delimited skill vocabulary."""
    skills = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            skill = line.strip()
            if skill:
                skills.append(skill)
    return skills


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_known_skills(text: str, skill_vocabulary: Iterable[str]) -> list[str]:
    """Return vocabulary skills that appear in the supplied text.

    Matching is case-insensitive and uses token boundaries so short skills such
    as R or Go do not match arbitrary substrings inside other words.
    """
    normalized_text = _normalize(text)
    matches: list[str] = []

    for skill in skill_vocabulary:
        normalized_skill = _normalize(skill)
        pattern = rf"(?<!\w){re.escape(normalized_skill)}(?!\w)"
        if re.search(pattern, normalized_text):
            matches.append(skill)

    # Preserve vocabulary order while removing duplicates.
    return list(dict.fromkeys(matches))

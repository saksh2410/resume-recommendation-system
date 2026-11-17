"""Domain skill-vocabulary loading and text matching."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_skill_vocabulary(path: str | Path) -> list[str]:
    """Load a newline-delimited skill vocabulary."""
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Skill vocabulary not found: {vocab_path}")

    skills: list[str] = []
    seen: set[str] = set()

    with vocab_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            skill = _normalize_whitespace(line)
            if not skill:
                continue

            key = skill.casefold()
            if key not in seen:
                seen.add(key)
                skills.append(skill)

    if not skills:
        raise ValueError(f"Skill vocabulary is empty: {vocab_path}")

    return skills


def extract_known_skills(
    text: str,
    skill_vocabulary: Iterable[str],
) -> list[str]:
    """Extract vocabulary skills present in text."""
    normalized_text = _normalize_whitespace(text)
    matches: list[str] = []
    seen: set[str] = set()

    for raw_skill in skill_vocabulary:
        skill = _normalize_whitespace(raw_skill)
        if not skill:
            continue

        key = skill.casefold()
        if key in seen:
            continue

        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            matches.append(raw_skill)
            seen.add(key)

    return matches

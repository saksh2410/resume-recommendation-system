"""Skill vocabulary loading and candidate extraction."""

from pathlib import Path
from typing import Iterable


def load_skill_vocabulary(path: str | Path) -> list[str]:
    """Load the project's newline-delimited skill vocabulary."""
    raise NotImplementedError


def extract_known_skills(
    text: str,
    skill_vocabulary: Iterable[str],
) -> list[str]:
    """Extract skill candidates from text using the supplied vocabulary."""
    raise NotImplementedError

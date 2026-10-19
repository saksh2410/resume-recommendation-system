"""Particle Swarm Optimization for JD-aware skill selection."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    """State for one binary PSO particle."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float


class BinaryPSOSkillSelector:
    """Select a JD-relevant subset of skills using Binary PSO."""

    def __init__(self, **kwargs) -> None:
        """Store PSO hyperparameters."""
        raise NotImplementedError

    def fit(
        self,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> "BinaryPSOSkillSelector":
        """Optimize the skill-selection mask."""
        raise NotImplementedError

    def transform(self, candidate_skills: list[str]) -> list[str]:
        """Return the selected skill subset."""
        raise NotImplementedError

    def fit_transform(
        self,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> list[str]:
        """Fit the optimizer and return selected skills."""
        raise NotImplementedError

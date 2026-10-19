"""Genetic Algorithm for resume-subset selection."""

from dataclasses import dataclass


@dataclass
class GAConfig:
    """Hyperparameters for the Genetic Algorithm."""

    population_size: int = 30
    generations: int = 10
    mutation_rate: float = 0.1
    recommendation_size: int = 5
    random_state: int = 42


class GeneticResumeSelector:
    """Select a group of resumes using a Genetic Algorithm."""

    def __init__(self, config: GAConfig | None = None) -> None:
        """Initialize the optimizer."""
        raise NotImplementedError

    def fit(self, scores: list[float]) -> "GeneticResumeSelector":
        """Optimize the resume subset against precomputed scores."""
        raise NotImplementedError

    def get_selected_indices(self) -> list[int]:
        """Return selected resume indices after fitting."""
        raise NotImplementedError

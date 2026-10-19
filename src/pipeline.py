"""End-to-end resume recommendation pipeline."""

from dataclasses import dataclass


@dataclass
class Recommendation:
    """One ranked resume recommendation."""

    resume_name: str
    score: float


class ResumeRecommendationPipeline:
    """Coordinate parsing, skill extraction, PSO, similarity, and GA."""

    def __init__(self, **kwargs) -> None:
        """Initialize pipeline dependencies and configuration."""
        raise NotImplementedError

    def fit(
        self,
        resume_directory: str,
        job_description_path: str,
    ) -> "ResumeRecommendationPipeline":
        """Build all intermediate representations for one JD."""
        raise NotImplementedError

    def recommend(self, top_n: int = 5) -> list[Recommendation]:
        """Return the final ranked resume recommendations."""
        raise NotImplementedError

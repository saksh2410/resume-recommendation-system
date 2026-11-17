"""Tests for Genetic Algorithm resume selection."""

from src.genetic_algorithm import (
    GAConfig,
    GeneticResumeSelector,
)


def test_ga_selects_highest_scoring_subset() -> None:
    selector = GeneticResumeSelector(
        GAConfig(
            population_size=30,
            generations=10,
            mutation_rate=0.1,
            recommendation_size=2,
            random_state=4,
        )
    )

    selector.fit([0.1, 0.9, 0.3, 0.8])

    assert selector.get_selected_indices() == [1, 3]
    assert selector.best_fitness_ == 1.7
    assert len(
        selector.average_fitness_history_
    ) == 10


def test_ga_handles_large_requested_subset() -> None:
    selector = GeneticResumeSelector(
        GAConfig(
            recommendation_size=5,
            random_state=1,
        )
    )

    selector.fit([0.2, 0.7])

    assert selector.get_selected_indices() == [1, 0]

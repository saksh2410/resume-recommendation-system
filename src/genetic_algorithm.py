"""Genetic Algorithm for course-required resume-subset selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
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
        self.config = config or GAConfig()

        if self.config.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if self.config.generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0.0 <= self.config.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.config.recommendation_size < 1:
            raise ValueError("recommendation_size must be at least 1")

        self.rng = random.Random(self.config.random_state)
        self.selected_indices_: list[int] | None = None
        self.best_fitness_: float | None = None
        self.average_fitness_history_: list[float] = []
        self.best_fitness_history_: list[float] = []

    @staticmethod
    def _fitness(individual: tuple[int, ...], scores: list[float]) -> float:
        return float(sum(scores[index] for index in individual))

    def _random_individual(
        self,
        n_resumes: int,
        size: int,
    ) -> tuple[int, ...]:
        return tuple(sorted(self.rng.sample(range(n_resumes), size)))

    def _initial_population(
        self,
        n_resumes: int,
        size: int,
    ) -> list[tuple[int, ...]]:
        max_unique = math.comb(n_resumes, size)
        target_size = min(self.config.population_size, max_unique)

        population: set[tuple[int, ...]] = set()
        while len(population) < target_size:
            population.add(self._random_individual(n_resumes, size))

        return list(population)

    def _select_parent(
        self,
        population: list[tuple[int, ...]],
        scores: list[float],
    ) -> tuple[int, ...]:
        fitnesses = [
            self._fitness(individual, scores)
            for individual in population
        ]
        minimum = min(fitnesses)
        weights = [fitness - minimum + 1e-12 for fitness in fitnesses]

        if sum(weights) <= 0:
            return self.rng.choice(population)

        return self.rng.choices(
            population,
            weights=weights,
            k=1,
        )[0]

    def _crossover(
        self,
        parent_a: tuple[int, ...],
        parent_b: tuple[int, ...],
        n_resumes: int,
    ) -> tuple[int, ...]:
        size = len(parent_a)

        if size == 1:
            child = [self.rng.choice((parent_a[0], parent_b[0]))]
        else:
            cut = self.rng.randint(1, size - 1)
            child = list(parent_a[:cut]) + list(parent_b[cut:])

        unique: list[int] = []
        for index in child:
            if index not in unique:
                unique.append(index)

        available = [
            index
            for index in range(n_resumes)
            if index not in unique
        ]
        self.rng.shuffle(available)
        unique.extend(available[: size - len(unique)])

        return tuple(sorted(unique))

    def _mutate(
        self,
        individual: tuple[int, ...],
        n_resumes: int,
    ) -> tuple[int, ...]:
        child = list(individual)

        for position in range(len(child)):
            if self.rng.random() >= self.config.mutation_rate:
                continue

            unavailable = set(child)
            choices = [
                index
                for index in range(n_resumes)
                if index not in unavailable
            ]
            if choices:
                child[position] = self.rng.choice(choices)

        return tuple(sorted(child))

    def fit(self, scores: list[float]) -> "GeneticResumeSelector":
        """Optimize a resume subset against precomputed similarity scores."""
        if not scores:
            raise ValueError("scores cannot be empty")

        numeric_scores = [float(score) for score in scores]
        n_resumes = len(numeric_scores)
        size = min(self.config.recommendation_size, n_resumes)

        population = self._initial_population(n_resumes, size)
        population_size = len(population)

        self.average_fitness_history_ = []
        self.best_fitness_history_ = []

        for _ in range(self.config.generations):
            fitnesses = [
                self._fitness(individual, numeric_scores)
                for individual in population
            ]
            self.average_fitness_history_.append(
                float(sum(fitnesses) / len(fitnesses))
            )
            self.best_fitness_history_.append(float(max(fitnesses)))

            offspring: set[tuple[int, ...]] = set()
            attempts = 0
            max_attempts = max(100, population_size * 20)

            while (
                len(offspring) < population_size
                and attempts < max_attempts
            ):
                parent_a = self._select_parent(
                    population,
                    numeric_scores,
                )
                parent_b = self._select_parent(
                    population,
                    numeric_scores,
                )

                child = self._crossover(
                    parent_a,
                    parent_b,
                    n_resumes,
                )
                child = self._mutate(child, n_resumes)
                offspring.add(child)
                attempts += 1

            combined = set(population) | offspring
            population = sorted(
                combined,
                key=lambda individual: (
                    -self._fitness(individual, numeric_scores),
                    individual,
                ),
            )[:population_size]

        best = max(
            population,
            key=lambda individual: self._fitness(
                individual,
                numeric_scores,
            ),
        )

        self.selected_indices_ = sorted(
            best,
            key=lambda index: (-numeric_scores[index], index),
        )
        self.best_fitness_ = self._fitness(best, numeric_scores)
        return self

    def get_selected_indices(self) -> list[int]:
        """Return selected resume indices ordered by descending score."""
        if self.selected_indices_ is None:
            raise RuntimeError(
                "fit must be called before get_selected_indices"
            )
        return list(self.selected_indices_)

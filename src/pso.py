"""Binary Particle Swarm Optimization for JD-aware skill selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    """State for one binary PSO particle."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float


@dataclass(frozen=True)
class PSOConfig:
    """Hyperparameters for Binary PSO."""

    n_particles: int = 30
    max_iter: int = 40
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    recall_weight: float = 0.6
    precision_weight: float = 0.4
    sparsity_penalty: float = 0.02
    random_state: int | None = 42


class BinaryPSOSkillSelector:
    """Select a JD-relevant subset of resume skills using Binary PSO."""

    def __init__(self, config: PSOConfig | None = None, **overrides) -> None:
        base = config or PSOConfig()
        values = {
            field: getattr(base, field)
            for field in PSOConfig.__dataclass_fields__
        }
        values.update(overrides)
        self.config = PSOConfig(**values)

        if self.config.n_particles < 1:
            raise ValueError("n_particles must be at least 1")
        if self.config.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.config.recall_weight < 0 or self.config.precision_weight < 0:
            raise ValueError("fitness weights must be non-negative")
        if self.config.sparsity_penalty < 0:
            raise ValueError("sparsity_penalty must be non-negative")

        self.rng = np.random.default_rng(self.config.random_state)
        self.best_position_: np.ndarray | None = None
        self.best_fitness_: float | None = None
        self.fitness_history_: list[float] = []
        self._candidate_count: int | None = None

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        return " ".join(skill.casefold().split())

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _fitness(
        self,
        position: np.ndarray,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> float:
        selected = [
            skill
            for skill, keep in zip(candidate_skills, position, strict=True)
            if keep
        ]

        if not selected:
            return 0.0

        selected_set = {self._normalize_skill(skill) for skill in selected}
        jd_set = {self._normalize_skill(skill) for skill in jd_skills}

        overlap = len(selected_set & jd_set)
        recall = overlap / len(jd_set) if jd_set else 0.0
        precision = overlap / len(selected_set) if selected_set else 0.0
        density = len(selected_set) / len(candidate_skills)

        return float(
            self.config.recall_weight * recall
            + self.config.precision_weight * precision
            - self.config.sparsity_penalty * density
        )

    def fit(
        self,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> "BinaryPSOSkillSelector":
        """Optimize and retain a binary skill-selection mask."""
        candidate_skills = list(dict.fromkeys(candidate_skills))
        jd_skills = list(dict.fromkeys(jd_skills))

        if not candidate_skills:
            raise ValueError("candidate_skills cannot be empty")
        if not jd_skills:
            raise ValueError("jd_skills cannot be empty")

        dim = len(candidate_skills)
        self._candidate_count = dim

        particles: list[Particle] = []
        global_best_position = np.zeros(dim, dtype=int)
        global_best_fitness = self._fitness(
            global_best_position, candidate_skills, jd_skills
        )

        for _ in range(self.config.n_particles):
            position = self.rng.integers(0, 2, size=dim, dtype=int)
            velocity = self.rng.uniform(-1.0, 1.0, size=dim)
            fitness = self._fitness(position, candidate_skills, jd_skills)

            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=fitness,
            )
            particles.append(particle)

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = position.copy()

        self.fitness_history_ = [float(global_best_fitness)]

        for _ in range(self.config.max_iter):
            for particle in particles:
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)

                particle.velocity = (
                    self.config.inertia * particle.velocity
                    + self.config.cognitive
                    * r1
                    * (particle.best_position - particle.position)
                    + self.config.social
                    * r2
                    * (global_best_position - particle.position)
                )

                probabilities = self._sigmoid(particle.velocity)
                particle.position = (
                    self.rng.random(dim) < probabilities
                ).astype(int)

                fitness = self._fitness(
                    particle.position, candidate_skills, jd_skills
                )

                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()

            self.fitness_history_.append(float(global_best_fitness))

        self.best_position_ = global_best_position
        self.best_fitness_ = float(global_best_fitness)
        return self

    def transform(self, candidate_skills: list[str]) -> list[str]:
        """Return the selected subset using the fitted best mask."""
        if self.best_position_ is None or self._candidate_count is None:
            raise RuntimeError("fit must be called before transform")
        if len(candidate_skills) != self._candidate_count:
            raise ValueError(
                "candidate_skills must match the list used during fit"
            )

        return [
            skill
            for skill, keep in zip(
                candidate_skills,
                self.best_position_,
                strict=True,
            )
            if keep
        ]

    def fit_transform(
        self,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> list[str]:
        """Fit the optimizer and return selected skills."""
        return self.fit(candidate_skills, jd_skills).transform(candidate_skills)

"""Binary Particle Swarm Optimization for JD-aware skill selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Particle:
    """State for one binary PSO particle."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float


class BinaryPSOSkillSelector:
    """Select a compact subset of resume skills relevant to a job description.

    A particle is a binary vector over the candidate skills found in a resume.
    A 1 means that skill is selected; 0 means it is excluded.

    Fitness rewards:
      * lexical similarity between the selected skill set and the JD skill set;
      * coverage of skills shared with the JD.

    A small sparsity penalty discourages selecting every candidate skill.
    """

    def __init__(
        self,
        n_particles: int = 30,
        max_iter: int = 40,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        similarity_weight: float = 0.7,
        coverage_weight: float = 0.3,
        sparsity_penalty: float = 0.05,
        random_state: int | None = 42,
    ) -> None:
        if n_particles < 1:
            raise ValueError("n_particles must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        self.n_particles = n_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.similarity_weight = similarity_weight
        self.coverage_weight = coverage_weight
        self.sparsity_penalty = sparsity_penalty
        self.rng = np.random.default_rng(random_state)

        self.best_position_: np.ndarray | None = None
        self.best_fitness_: float | None = None
        self.fitness_history_: list[float] = []

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -30, 30)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        return " ".join(skill.lower().split())

    def _fitness(
        self,
        position: np.ndarray,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> float:
        selected = [
            skill for skill, keep in zip(candidate_skills, position, strict=True) if keep
        ]
        if not selected:
            return -1.0

        selected_text = " ".join(selected)
        jd_text = " ".join(jd_skills)

        vectorizer = TfidfVectorizer(lowercase=True)
        vectors = vectorizer.fit_transform([selected_text, jd_text])
        similarity = float(cosine_similarity(vectors[0], vectors[1])[0, 0])

        jd_normalized = {self._normalize_skill(skill) for skill in jd_skills}
        selected_normalized = {self._normalize_skill(skill) for skill in selected}

        if jd_normalized:
            coverage = len(selected_normalized & jd_normalized) / len(jd_normalized)
        else:
            coverage = 0.0

        density = len(selected) / len(candidate_skills)

        return (
            self.similarity_weight * similarity
            + self.coverage_weight * coverage
            - self.sparsity_penalty * density
        )

    def fit(
        self,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> "BinaryPSOSkillSelector":
        """Run binary PSO and retain the best skill-selection mask."""
        candidate_skills = list(dict.fromkeys(candidate_skills))
        jd_skills = list(dict.fromkeys(jd_skills))

        if not candidate_skills:
            raise ValueError("candidate_skills cannot be empty")
        if not jd_skills:
            raise ValueError("jd_skills cannot be empty")

        dim = len(candidate_skills)

        particles: list[Particle] = []
        global_best_position: np.ndarray | None = None
        global_best_fitness = -np.inf

        for _ in range(self.n_particles):
            position = self.rng.integers(0, 2, size=dim, dtype=int)
            if not position.any():
                position[self.rng.integers(0, dim)] = 1

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

        for _ in range(self.max_iter):
            assert global_best_position is not None

            for particle in particles:
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)

                particle.velocity = (
                    self.inertia * particle.velocity
                    + self.cognitive
                    * r1
                    * (particle.best_position - particle.position)
                    + self.social
                    * r2
                    * (global_best_position - particle.position)
                )

                probabilities = self._sigmoid(particle.velocity)
                particle.position = (
                    self.rng.random(dim) < probabilities
                ).astype(int)

                if not particle.position.any():
                    particle.position[self.rng.integers(0, dim)] = 1

                fitness = self._fitness(
                    particle.position,
                    candidate_skills,
                    jd_skills,
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
        """Return the skills selected by the fitted particle mask."""
        if self.best_position_ is None:
            raise RuntimeError("fit must be called before transform")

        if len(candidate_skills) != len(self.best_position_):
            raise ValueError(
                "candidate_skills must match the candidate list used during fit"
            )

        return [
            skill
            for skill, keep in zip(candidate_skills, self.best_position_, strict=True)
            if keep
        ]

    def fit_transform(
        self,
        candidate_skills: list[str],
        jd_skills: list[str],
    ) -> list[str]:
        """Fit PSO and return the selected subset."""
        self.fit(candidate_skills, jd_skills)
        return self.transform(candidate_skills)

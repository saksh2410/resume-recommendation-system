"""Tests for Binary PSO skill selection."""

import numpy as np

from src.pso import BinaryPSOSkillSelector, PSOConfig


def test_pso_selects_jd_relevant_skills() -> None:
    candidates = [
        "Python",
        "machine learning",
        "Docker",
        "graphic design",
        "sales",
    ]
    jd_skills = [
        "Python",
        "machine learning",
        "Docker",
    ]

    selector = BinaryPSOSkillSelector(
        config=PSOConfig(
            n_particles=50,
            max_iter=60,
            random_state=7,
        )
    )
    selected = selector.fit_transform(
        candidates,
        jd_skills,
    )

    assert set(selected) == {
        "Python",
        "machine learning",
        "Docker",
    }
    assert selector.best_fitness_ is not None
    assert np.isfinite(selector.best_fitness_)
    assert (
        selector.fitness_history_[-1]
        >= selector.fitness_history_[0]
    )


def test_pso_can_select_nothing_without_overlap() -> None:
    selector = BinaryPSOSkillSelector(
        config=PSOConfig(
            n_particles=20,
            max_iter=20,
            random_state=2,
        )
    )

    selected = selector.fit_transform(
        ["sales", "graphic design"],
        ["Python", "machine learning"],
    )

    assert selected == []

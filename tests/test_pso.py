"""Tests for reconstructed binary PSO skill selection."""

import numpy as np

from src.pso import BinaryPSOSkillSelector
from src.pso_pipeline import select_resume_skills_with_pso
from src.skills import extract_known_skills


def test_extract_known_skills_uses_boundaries() -> None:
    skills = ["Python", "R", "Go", "project management"]
    text = "Python and project management are core skills. Growing is not Go."

    extracted = extract_known_skills(text, skills)

    assert "Python" in extracted
    assert "project management" in extracted
    assert "R" not in extracted


def test_pso_prefers_jd_relevant_skills() -> None:
    candidates = [
        "Python",
        "machine learning",
        "Docker",
        "graphic design",
        "sales",
    ]
    jd_skills = ["Python", "machine learning", "Docker"]

    selector = BinaryPSOSkillSelector(
        n_particles=40,
        max_iter=60,
        sparsity_penalty=0.08,
        random_state=7,
    )
    selected = selector.fit_transform(candidates, jd_skills)

    assert "Python" in selected
    assert "machine learning" in selected
    assert len(set(selected) & {"Python", "machine learning", "Docker"}) >= 2
    assert selector.best_fitness_ is not None
    assert np.isfinite(selector.best_fitness_)


def test_pso_pipeline_returns_candidate_and_selected_skills() -> None:
    vocabulary = [
        "Python",
        "SQL",
        "machine learning",
        "project management",
        "graphic design",
    ]
    resume = "Python, SQL, machine learning and graphic design."
    jd = "Looking for Python and machine learning experience."

    candidates, jd_skills, selected, selector = select_resume_skills_with_pso(
        resume,
        jd,
        vocabulary,
        random_state=3,
    )

    assert candidates == ["Python", "SQL", "machine learning", "graphic design"]
    assert jd_skills == ["Python", "machine learning"]
    assert set(selected).issubset(set(candidates))
    assert selector.best_position_ is not None

"""Tests for the end-to-end recommendation pipeline."""

from pathlib import Path

from src.genetic_algorithm import GAConfig
from src.pipeline import ResumeRecommendationPipeline
from src.pso import PSOConfig


def test_pipeline_integrates_all_stages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    vocab = tmp_path / "it_skills.txt"
    vocab.write_text(
        (
            "Python\n"
            "machine learning\n"
            "Docker\n"
            "graphic design\n"
            "sales\n"
        ),
        encoding="utf-8",
    )

    resumes = {
        "ml_engineer.pdf": (
            "Python machine learning Docker"
        ),
        "designer.pdf": "graphic design",
        "hybrid.pdf": "Python sales",
    }
    jd = (
        "We need Python, machine learning "
        "and Docker experience."
    )

    monkeypatch.setattr(
        "src.pipeline.load_resume_directory",
        lambda _: resumes,
    )
    monkeypatch.setattr(
        "src.pipeline.load_job_description",
        lambda _: jd,
    )

    pipeline = ResumeRecommendationPipeline(
        skill_vocab_path=vocab,
        pso_config=PSOConfig(
            n_particles=40,
            max_iter=50,
            random_state=3,
        ),
        ga_config=GAConfig(
            population_size=20,
            generations=8,
            recommendation_size=2,
            random_state=3,
        ),
    )

    pipeline.fit(
        "unused",
        "unused.pdf",
    )
    recommendations = pipeline.recommend()

    assert recommendations[0].resume_name == (
        "ml_engineer.pdf"
    )
    assert (
        recommendations[0].score
        > recommendations[1].score
    )
    assert pipeline.jd_skills_ == [
        "Python",
        "machine learning",
        "Docker",
    ]
    assert (
        pipeline.selected_skills_["designer.pdf"]
        == []
    )

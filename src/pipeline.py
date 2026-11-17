"""End-to-end resume recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .genetic_algorithm import GAConfig, GeneticResumeSelector
from .parser import load_job_description, load_resume_directory
from .pso import BinaryPSOSkillSelector, PSOConfig
from .similarity import (
    build_tfidf_matrix,
    cosine_scores_against_target,
)
from .skills import extract_known_skills, load_skill_vocabulary


@dataclass(frozen=True)
class Recommendation:
    """One ranked resume recommendation."""

    resume_name: str
    score: float


class ResumeRecommendationPipeline:
    """Coordinate parsing, skill extraction, PSO, similarity, and GA."""

    def __init__(
        self,
        skill_vocab_path: str | Path = config.IT_SKILL_VOCAB_PATH,
        pso_config: PSOConfig | None = None,
        ga_config: GAConfig | None = None,
    ) -> None:
        self.skill_vocab_path = Path(skill_vocab_path)
        self.pso_config = pso_config or PSOConfig(
            random_state=config.DEFAULT_RANDOM_STATE
        )
        self.ga_config = ga_config or GAConfig(
            recommendation_size=config.DEFAULT_RECOMMENDATION_SIZE,
            random_state=config.DEFAULT_RANDOM_STATE,
        )

        self.resume_names_: list[str] = []
        self.resume_scores_: list[float] = []
        self.jd_skills_: list[str] = []
        self.candidate_skills_: dict[str, list[str]] = {}
        self.selected_skills_: dict[str, list[str]] = {}
        self.ga_selector_: GeneticResumeSelector | None = None
        self._fitted = False

    def _pso_config_for_resume(self, offset: int) -> PSOConfig:
        values = {
            field: getattr(self.pso_config, field)
            for field in PSOConfig.__dataclass_fields__
        }

        if self.pso_config.random_state is not None:
            values["random_state"] = (
                self.pso_config.random_state + offset
            )

        return PSOConfig(**values)

    def fit(
        self,
        resume_directory: str | Path,
        job_description_path: str | Path,
    ) -> "ResumeRecommendationPipeline":
        """Build all intermediate representations for one JD."""
        resumes = load_resume_directory(resume_directory)
        job_description = load_job_description(job_description_path)
        skill_vocabulary = load_skill_vocabulary(
            self.skill_vocab_path
        )

        jd_skills = extract_known_skills(
            job_description,
            skill_vocabulary,
        )
        if not jd_skills:
            raise ValueError(
                "No skills from the selected vocabulary were found "
                "in the job description"
            )

        self.resume_names_ = list(resumes.keys())
        self.jd_skills_ = jd_skills
        self.candidate_skills_ = {}
        self.selected_skills_ = {}

        for offset, (resume_name, resume_text) in enumerate(
            resumes.items()
        ):
            candidates = extract_known_skills(
                resume_text,
                skill_vocabulary,
            )
            self.candidate_skills_[resume_name] = candidates

            if not candidates:
                self.selected_skills_[resume_name] = []
                continue

            selector = BinaryPSOSkillSelector(
                config=self._pso_config_for_resume(offset)
            )
            self.selected_skills_[resume_name] = (
                selector.fit_transform(candidates, jd_skills)
            )

        documents = [" ".join(jd_skills)] + [
            " ".join(self.selected_skills_[name])
            for name in self.resume_names_
        ]
        matrix = build_tfidf_matrix(documents)
        all_scores = cosine_scores_against_target(
            matrix,
            target_index=0,
        )
        self.resume_scores_ = all_scores[1:]

        self.ga_selector_ = GeneticResumeSelector(self.ga_config)
        self.ga_selector_.fit(self.resume_scores_)

        self._fitted = True
        return self

    def recommend(
        self,
        top_n: int | None = None,
    ) -> list[Recommendation]:
        """Return GA-selected resumes ranked by similarity score."""
        if not self._fitted or self.ga_selector_ is None:
            raise RuntimeError("fit must be called before recommend")

        selected_indices = self.ga_selector_.get_selected_indices()

        if top_n is None:
            top_n = len(selected_indices)
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        return [
            Recommendation(
                resume_name=self.resume_names_[index],
                score=self.resume_scores_[index],
            )
            for index in selected_indices[:top_n]
        ]

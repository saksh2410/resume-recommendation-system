"""Utilities connecting skill extraction to binary PSO."""

from __future__ import annotations

from .pso import BinaryPSOSkillSelector
from .skills import extract_known_skills


def select_resume_skills_with_pso(
    resume_text: str,
    job_description_text: str,
    skill_vocabulary: list[str],
    *,
    random_state: int | None = 42,
) -> tuple[list[str], list[str], list[str], BinaryPSOSkillSelector]:
    """Extract candidate skills and select the JD-relevant subset with PSO.

    Returns:
        candidate_skills: all known skills detected in the resume.
        jd_skills: all known skills detected in the job description.
        selected_skills: PSO-selected resume skill subset.
        selector: fitted BinaryPSOSkillSelector instance.
    """
    candidate_skills = extract_known_skills(resume_text, skill_vocabulary)
    jd_skills = extract_known_skills(job_description_text, skill_vocabulary)

    if not candidate_skills:
        raise ValueError("No known skills were found in the resume")
    if not jd_skills:
        raise ValueError("No known skills were found in the job description")

    selector = BinaryPSOSkillSelector(random_state=random_state)
    selected_skills = selector.fit_transform(candidate_skills, jd_skills)

    return candidate_skills, jd_skills, selected_skills, selector

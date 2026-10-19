"""Project configuration and shared paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESUME_DIR = DATA_DIR / "resumes"
JOB_DESCRIPTION_DIR = DATA_DIR / "job_descriptions"
SKILL_VOCAB_DIR = DATA_DIR / "jd_skills"
IT_SKILL_VOCAB_PATH = SKILL_VOCAB_DIR / "it_skills.txt"

DEFAULT_RECOMMENDATION_SIZE = 5
DEFAULT_RANDOM_STATE = 42

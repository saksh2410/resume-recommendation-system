"""Resume and job-description parsing utilities."""

from pathlib import Path


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract plain text from a PDF file."""
    raise NotImplementedError


def load_resume_directory(directory: str | Path) -> dict[str, str]:
    """Load all resume PDFs in a directory as {filename: text}."""
    raise NotImplementedError


def load_job_description(path: str | Path) -> str:
    """Load and parse a job-description PDF."""
    raise NotImplementedError

"""Resume and job-description PDF parsing utilities."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract plain text from a PDF file."""
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(page for page in pages if page).strip()


def load_resume_directory(directory: str | Path) -> dict[str, str]:
    """Load all PDF resumes in a directory as filename-to-text mappings."""
    resume_dir = Path(directory)

    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
    if not resume_dir.is_dir():
        raise NotADirectoryError(f"Resume path is not a directory: {resume_dir}")

    pdf_files = sorted(
        (
            path
            for path in resume_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )

    if not pdf_files:
        raise ValueError(f"No PDF resumes found in: {resume_dir}")

    return {path.name: extract_text_from_pdf(path) for path in pdf_files}


def load_job_description(path: str | Path) -> str:
    """Load and parse a job-description PDF."""
    return extract_text_from_pdf(path)

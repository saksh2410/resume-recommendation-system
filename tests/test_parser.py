"""Tests for PDF parsing utilities."""

from pathlib import Path

import pytest

from src import parser


def test_extract_text_requires_existing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        parser.extract_text_from_pdf(
            tmp_path / "missing.pdf"
        )


def test_extract_text_rejects_non_pdf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resume.txt"
    path.write_text("resume", encoding="utf-8")

    with pytest.raises(ValueError):
        parser.extract_text_from_pdf(path)


def test_load_resume_directory_is_sorted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "z_resume.pdf").touch()
    (tmp_path / "A_resume.pdf").touch()
    (tmp_path / "notes.txt").touch()

    monkeypatch.setattr(
        parser,
        "extract_text_from_pdf",
        lambda path: f"text:{Path(path).name}",
    )

    resumes = parser.load_resume_directory(tmp_path)

    assert list(resumes) == [
        "A_resume.pdf",
        "z_resume.pdf",
    ]

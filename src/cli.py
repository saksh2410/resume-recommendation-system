"""Command-line interface for the resume recommendation system."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .genetic_algorithm import GAConfig
from .pipeline import ResumeRecommendationPipeline
from .pso import PSOConfig


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank PDF resumes against a target job description."
        )
    )
    parser.add_argument(
        "job_description",
        type=Path,
        help="Path to the target job-description PDF.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=config.RESUME_DIR,
        help="Directory containing PDF resumes.",
    )
    parser.add_argument(
        "--skill-vocab",
        type=Path,
        default=config.IT_SKILL_VOCAB_PATH,
        help="Domain-specific newline-delimited skill vocabulary.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=config.DEFAULT_RECOMMENDATION_SIZE,
        help="Number of resumes to select and return.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_RANDOM_STATE,
        help="Random seed for PSO and GA.",
    )
    return parser


def main() -> None:
    """Run the recommendation pipeline and print ranked results."""
    args = build_parser().parse_args()

    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")

    pipeline = ResumeRecommendationPipeline(
        skill_vocab_path=args.skill_vocab,
        pso_config=PSOConfig(random_state=args.seed),
        ga_config=GAConfig(
            recommendation_size=args.top_n,
            random_state=args.seed,
        ),
    )
    pipeline.fit(
        args.resume_dir,
        args.job_description,
    )

    recommendations = pipeline.recommend(args.top_n)

    print("\nRecommended resumes:")
    for rank, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"{rank:>2}. {recommendation.resume_name} "
            f"(similarity={recommendation.score:.4f})"
        )


if __name__ == "__main__":
    main()
